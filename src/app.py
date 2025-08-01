import streamlit as st
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace, Literal, OWL, RDF, URIRef
from pyvis.network import Network
from pathlib import Path
import tempfile, re, unicodedata, os, json

# Función para normalizar columnas igual que en el backend
def normalize_column(col):
    return re.sub(r"[^a-zA-Z0-9_]", "", col.strip().replace(" ", "_")).lower()

# Limpieza de segmentos para URIs
def clean_uri_segment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    text = str(text)
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^\w-]', '', text.replace(" ", "-"))
    return text.lower()

st.set_page_config(page_title="CSV a RDF", page_icon="🧬", layout="wide")

st.markdown("""
    <h1 style="text-align:center;">📄 C2R 🧬</h1>
    <h3 style="text-align:center; ">Convierte tus datos tabulares en conocimiento enlazado</h3>
    <div style="text-align:center; margin-bottom: 20px;">
    </div>
""", unsafe_allow_html=True)

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

uploaded_file = st.file_uploader("Sube tu archivo CSV:", type=["csv"])

if uploaded_file:
    # Limpiar caché si cambia archivo
    if st.session_state.last_file_name != uploaded_file.name:
        st.session_state.mapping = None
        st.session_state.last_file_name = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        csv_path = tmp.name

    df = pd.read_csv(csv_path)
    st.subheader("Vista previa del CSV")
    st.dataframe(df.head(), use_container_width=True)

    # Normalizar columnas
    df.columns = [normalize_column(c) for c in df.columns]

    # ------ MAPEADO CON IA ------
    st.subheader("🔎 Mapea tus columnas con IA (Gemini)")
    if st.button("Mapear", key="mapear_btn", help="Haz clic para que la IA analice las columnas y sugiera predicados."):
        # Aquí se importa tu función mapear_predicados
        from algoritmo import mapear_predicados
        mapping = mapear_predicados(df, uploaded_file.name)
        st.session_state.mapping = mapping
        if not os.path.exists("./salida"):
            os.makedirs("./salida")
        pd.DataFrame(list(mapping.items()), columns=["Columna", "Predicado (URI)"]).to_csv(
            "./salida/mapping_generado.csv", index=False
        )
        st.success("✅ ¡Mapeo realizado con éxito!")
    else:
        mapping = st.session_state.get("mapping", None)

    # Permitir cargar mapping desde archivo generado
    if mapping is None and os.path.exists("./salida/mapping_generado.csv"):
        mapping_df = pd.read_csv("./salida/mapping_generado.csv")
        mapping = dict(zip(mapping_df["Columna"], mapping_df["Predicado (URI)"]))
        st.session_state.mapping = mapping

    if mapping:
        st.table(pd.DataFrame(list(mapping.items()), columns=["Columna", "Predicado Sugerido (URI)"]))

    # ------ SELECCIÓN DE SUJETOS ------
    st.subheader("🔑 Selección de columna(s) que identifican cada fila")
    id_columns = st.multiselect(
        "✨ ¿Cuál o cuáles columnas forman el 'sujeto' en tu CSV?",
        options=list(df.columns),
        default=[df.columns[0]]
    )

    # ------ GENERACIÓN DE RDF Y VISUALIZACIÓN ------
    cols = st.columns([2, 1, 2])
    with cols[1]:  # columna central
        generar = st.button("🚀 Generar RDF y Grafo", use_container_width=True)
    if generar:
        SCHEMA = Namespace("http://schema.org/")
        g = Graph()
        g.namespace_manager.bind("schema", SCHEMA, override=True)
        g.namespace_manager.bind("owl", OWL)
        g.namespace_manager.bind("rdf", RDF)

        archivo_base = clean_uri_segment(Path(uploaded_file.name).stem)
        namespace_dinamico = Namespace(f"http://{archivo_base}/")
        g.namespace_manager.bind(archivo_base, namespace_dinamico)

        for idx, row in df.iterrows():
            pub_id = "_".join([str(row[col]) for col in id_columns if col in row and pd.notna(row[col])])
            if not pub_id:
                pub_id = f"row-{idx+1}"
            res_uri = namespace_dinamico[f"resource_{clean_uri_segment(pub_id)}"]
            g.add((res_uri, RDF.type, OWL.Thing))

            for col, val in row.items():
                if pd.isna(val) or str(val).strip() == "":
                    continue
                # ---> ¡USAR EL PREDICADO SUGERIDO POR GEMINI!
                pred_uri = mapping.get(col, col.lower())
                pred = URIRef(pred_uri) if isinstance(pred_uri, str) and pred_uri.startswith("http") else namespace_dinamico[col.lower()]

                # --- MANEJO ROBUSTO DE LISTAS Y JSON ---
                # Si ya es lista Python
                if isinstance(val, list):
                    for item in val:
                        g.add((res_uri, pred, Literal(item)))
                    continue

                # Si es string de lista por "," o ";"
                if isinstance(val, str) and ("," in val or ";" in val):
                    delim = "," if val.count(",") >= val.count(";") else ";"
                    items = [i.strip() for i in val.split(delim) if i.strip()]
                    if len(items) > 1:
                        for item in items:
                            g.add((res_uri, pred, Literal(item)))
                        continue

                # Si es JSON válido
                if isinstance(val, str):
                    try:
                        obj = json.loads(val)
                        if isinstance(obj, list):
                            for item in obj:
                                g.add((res_uri, pred, Literal(str(item))))
                            continue
                        elif isinstance(obj, dict):
                            g.add((res_uri, pred, Literal(json.dumps(obj))))
                            continue
                    except Exception:
                        pass

                # Valor normal único
                g.add((res_uri, pred, Literal(val)))

        ttl_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ttl")
        g.serialize(destination=ttl_file.name, format="turtle")
        st.success(f"✅ RDF generado con **{len(g)} tripletas**")

        st.subheader("🔍 Tripletas RDF generadas")
        triples = [{"Sujeto": str(s), "Predicado": str(p), "Objeto": str(o)} for s, p, o in g]
        st.dataframe(triples, use_container_width=True)

        with open(ttl_file.name, "rb") as f:
            st.download_button("⬇️ Descargar RDF (.ttl)", f, file_name="resultado.ttl")

        # -------- VISUALIZACIÓN DEL GRAFO --------
        st.subheader("🌐 Visualización del Grafo RDF")
        G = nx.Graph()
        for s, p, o in g:
            G.add_node(str(s))
            G.add_node(str(o))
            G.add_edge(str(s), str(o), label=str(p))

        node_colors = {}
        for node in G.nodes():
            if "resource_" in node:
                node_colors[node] = ("#1074A2", 25)
            elif node.startswith("http"):
                node_colors[node] = ("#A5B9C8", 18)
            else:
                node_colors[node] = ("#D7E3EB", 12)

        net = Network(height="700px", width="100%", directed=True, bgcolor="#f8f8f8")
        net.from_nx(G)
        for node in net.nodes:
            color, size = node_colors.get(node["id"], ("#b3b3b3", 10))
            node["color"] = color
            node["size"] = size
            node["title"] = node["id"]

        net.repulsion(node_distance=400, spring_length=150)
        html_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net.write_html(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)

else:
    st.info("⬆️ Sube un CSV para comenzar la conversión y visualización.")

# Forzar botones azules:
st.markdown("""
<style>
.stButton > button {
    background-color: #2471A3 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: bold;
    border: 1px solid #0d4663 !important;
}
div.stButton > button:hover {
    background-color: #1a5276 !important;
    color: #FFF !important;
}
</style>
""", unsafe_allow_html=True)
