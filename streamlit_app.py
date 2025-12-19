# streamlit_app.py
import os
import tempfile
import streamlit as st

from pdf_report import create_pdf_report

st.set_page_config(page_title="Soccer Central Quality Control", layout="wide")
st.title("Soccer Central Quality Control")

survey_type = st.selectbox("Survey type", ["players", "families"], index=0)
cycle_label = st.text_input("Cycle label", value="Cycle 3")
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if st.button("Run analysis"):
    if uploaded is None:
        st.error("Upload an Excel file first.")
    else:
        with st.spinner("Generating PDF..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            tmp.write(uploaded.getbuffer())
            tmp.close()

            pdf_path = create_pdf_report(
                input_path=tmp.name,
                cycle_label=cycle_label,
                survey_type=survey_type,
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF report",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )
            os.unlink(tmp.name)
