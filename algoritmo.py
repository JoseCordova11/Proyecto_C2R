import os
import pandas as pd
import re
import unicodedata
import json
from pathlib import Path
from rdflib import Namespace, Literal, URIRef, Graph, RDFS, DCTERMS, FOAF, SKOS, DC, OWL, XSD
import google.generativeai as genai
from deep_translator import GoogleTranslator

# =======================
# 1. CONFIGURACIÓN GLOBAL
# =======================

# --- Configuracion de API de Gemini ---
API_KEY = os.environ.get("GOOGLE_API_KEY") or "AIzaSyD9CmRluEHHrnY0i6FkmZhUivRoKio8hMU"
genai.configure(api_key=API_KEY)

# --- Lista de vocabularios RDF a utilizadas ---
VOCAB_URIS = {
    "foaf": "http://xmlns.com/foaf/spec/index.rdf",
    "bibo": "http://purl.org/ontology/bibo/bibo.rdf",
    "dcterms": "https://www.dublincore.org/2012/06/14/dcterms.ttl",               
    "skos": "http://www.w3.org/2009/08/skos-reference/skos.rdf"
}

SALIDA_DIR = "./salida"
if not os.path.exists(SALIDA_DIR):
    os.makedirs(SALIDA_DIR)

VOCAB_PROPS_PATH = os.path.join(SALIDA_DIR, "vocab_props.csv")

# ==============================
# 2. FUNCIONES DE VOCABULARIOS
# ==============================

def extraer_vocab_props():
    props = []
    for vocab, url in VOCAB_URIS.items():
        print(f"Descargando/procesando vocabulario: {vocab}")
        try:
            g = Graph()
            # Detecta archivos TTL y usa el parser adecuado
            if url.endswith(".ttl"):
                g.parse(url, format="turtle")
            else:
                g.parse(url)
            # Extrae todas las propiedades que tengan rdfs:comment
            for s, p, o in g.triples((None, RDFS.comment, None)):
                if (isinstance(s, URIRef) and isinstance(o, Literal)
                        and (str(s).startswith("http://") or str(s).startswith("https://"))):
                    # label: primero busca rdfs:label, si no existe usa fragmento de la URI
                    label = None
                    for _, _, label_o in g.triples((s, RDFS.label, None)):
                        label = str(label_o)
                        break
                    if not label:
                        label = str(s).split("/")[-1].split("#")[-1]
                    props.append({
                        "uri": str(s),
                        "label": label.lower(),
                        "desc": str(o)
                    })
        except Exception as e:
            print(f"Error procesando vocabulario {vocab}: {e}")
    return props

# --- Lee del cache o extrae los vocabularios ---
if os.path.exists(VOCAB_PROPS_PATH):
    VOCAB_PROPS = pd.read_csv(VOCAB_PROPS_PATH).to_dict(orient="records")
else:
    VOCAB_PROPS = extraer_vocab_props()
    pd.DataFrame(VOCAB_PROPS).to_csv(VOCAB_PROPS_PATH, index=False)

# ==================================
# 3. FUNCIONES AUXILIARES DE LIMPIEZA
# ==================================

def clean_uri_segment(text):
    # Normaliza cadenas para URIs: remueve acentos y caracteres especiales.
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    text = str(text)
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^\w-]', '', text.replace(" ", "-"))
    return text.lower()

def detect_language(text):
    #Detecta si el texto parece español o inglés.
    if isinstance(text, str) and re.search(r"[áéíóúñÁÉÍÓÚÑ]", text):
        return "es"
    return "en"

def clean_keyword(word):
    #Limpia palabras clave, remueve caracteres no alfanuméricos.
    if not isinstance(word, str):
        return word
    word = re.sub(r"^[^a-zA-Z0-9]+", "", word)
    word = word.replace("-", " ").strip()
    return word.lower()

def normalize_column(col):
    # Normaliza espacios y caracteres
    col = re.sub(r"[^a-zA-Z0-9_]", "", col.strip().replace(" ", "_"))
    return col.lower()

# ===================================
# 4. INFERENCIA DE TIPOS RDF LITERAL
# ===================================

def infer_literal(value, colname=None):
    if pd.isna(value) or str(value).strip() == "":
        return None

    val = str(value).strip()

    # Detecta y maneja JSON válido
    try:
        obj = json.loads(val)
        if isinstance(obj, list):
            return [Literal(str(item), lang=detect_language(str(item))) for item in obj]
        if isinstance(obj, dict):
            return Literal(json.dumps(obj), lang="en")
    except (json.JSONDecodeError, TypeError):
        pass

    # Detecta listas separadas por "," o ";"
    if ("," in val or ";" in val) and not val.startswith("http"):
        delim = "," if val.count(",") >= val.count(";") else ";"
        items = [v.strip() for v in val.split(delim) if v.strip() != ""]
        if len(items) > 1:
            return [Literal(item, lang=detect_language(item)) for item in items]

    if val.startswith("http"):
        return URIRef(val)
    if colname and colname.lower() in ["year", "date"] and re.fullmatch(r'\d{4}', val):
        return Literal(val, datatype=XSD.gYear)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
        return Literal(val, datatype=XSD.date)
    if val.lower() in ["true", "false", "yes", "no"]:
        return Literal(val.lower() in ["true", "yes"], datatype=XSD.boolean)
    try:
        if "." in val:
            num = float(val)
            if num.is_integer():
                return Literal(int(num), datatype=XSD.integer)
            return Literal(num, datatype=XSD.float)
        return Literal(int(val), datatype=XSD.integer)
    except ValueError:
        lang = detect_language(val)
        return Literal(val, lang=lang)


# ========================================
# 5. MAPEO DE COLUMNAS CON IA
# ========================================

def mapear_predicados(df, archivo_csv):
    import logging
    model = genai.GenerativeModel('gemini-1.5-flash')
    mapping = {}
    descripciones = []

    # Normaliza nombres de columnas
    df.columns = [normalize_column(c) for c in df.columns]

    # Genera namespace personalizado basado en el archivo
    archivo_base = clean_uri_segment(Path(archivo_csv).stem)
    namespace_dinamico = Namespace(f"http://{archivo_base}/")

    for columna in df.columns:
        # Toma algunos ejemplos de valores para el prompt
        
        ejemplos = df[columna].dropna().astype(str).unique().tolist()
        ejemplos_str = "; ".join(ejemplos[:3])
        desc_prompt = (
            f"Tengo una columna de datos llamada '{columna}'. "
            f"Algunos ejemplos de valores son: {ejemplos_str}. "
            "Describe con una frase clara qué representa esa columna."
        )
        # Usa Gemini para generar la descripción semántica
        try:
            response = model.generate_content(desc_prompt)
            descripcion = response.text.strip()
        except Exception as e:
            descripcion = columna
            logging.warning(f"Error al generar descripción con Gemini para columna {columna}: {e}")

        descripciones.append({"columna": columna, "descripcion": descripcion})

        # Prompt a Gemini para elegir el mejor predicado entre los vocabularios
        options = "\n".join([f"{prop['uri']} - {prop['desc']}" for prop in VOCAB_PROPS])
        match_prompt = (
            f"Tengo la siguiente descripción de una columna: '{descripcion}'.\n"
            "De la siguiente lista de propiedades RDF, elige SOLO UNA URI que mejor coincida y devuélveme SOLO la URI exacta. Si ninguna coincide, responde NONE.\n"
            f"{options}\n\n"
            "Recuerda: SOLO la URI exacta, NADA MÁS."
        )
        try:
            response2 = model.generate_content(match_prompt)
            uris = re.findall(r'http[s]?://[^\s\)\],]+', response2.text)
            best_uri = uris[0] if uris else None
        except Exception as e:
            best_uri = None
            logging.warning(f"Error al mapear predicado con Gemini para columna {columna}: {e}")
        # Si Gemini no sugiere una URI estándar, genera un predicado personalizado con el namespace del archivo
        if best_uri is None or (isinstance(best_uri, str) and best_uri.upper() == "NONE"):
            pred_uri = str(namespace_dinamico[columna.lower()])
        else:
            pred_uri = best_uri
        mapping[columna] = pred_uri

    # Guarda las descripciones de las columnas para auditoría
    descripciones_path = os.path.join(SALIDA_DIR, "descripciones.csv")
    pd.DataFrame(descripciones).to_csv(descripciones_path, index=False)

    return mapping
