"""
Integrated Retinal Vessel Analyzer - Gradio Frontend
Imports all analysis logic from vessel_analyzer.py
"""

import gradio as gr
from vessel_analyzer import run_full_analysis


# ================= GRADIO WRAPPER =================

def analyze_fundus_image(image):
    """
    Gradio callback – delegates to the backend and unpacks results
    into the positional tuple Gradio expects.
    """
    if image is None:
        return None, None, None, None, None, None, "Please upload an image first."

    results = run_full_analysis(image)

    return (
        results['mask_rgb_display'],
        results['artery_display'],
        results['vein_display'],
        results['artery_viz'],
        results['vein_viz'],
        results['json_output'],
        results['summary'],
    )


# ================= GRADIO INTERFACE =================

custom_css = """
    .input-section   {min-height: 400px;}
    .output-section  {min-height: 400px;}
    .analysis-section{min-height: 500px;}
    .summary-section {min-height: 400px; max-height: 400px; overflow-y: auto;}
    .json-section    {min-height: 400px; max-height: 400px; overflow-y: auto;}

    .main-title {
        text-align: center; padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    .main-title h1 {color: white !important; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);}
    .main-title p  {color: #f0f0f0 !important; margin: 5px 0 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);}

    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px; border-radius: 8px; margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header b {color: white !important; font-size: 16px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
    """


def create_gradio_interface():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Integrated Retinal Vessel Analyzer") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("""
            <div class="main-title">
            <h1>🔬 Integrated Retinal Vessel Analyzer</h1>
            <p>Automated fundus image segmentation and comprehensive vessel parameter analysis</p>
            </div>
            """)

        # ── Row 1: Input + Segmentation mask ────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>📤 INPUT Fundus Image</b></div>')
                input_image = gr.Image(
                    type="pil",
                    label="Upload Fundus Image",
                    height=350,
                    elem_classes="input-section"
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>🎨 Retinal AV Mask</b></div>')
                segmentation_output = gr.Image(
                    label="Full Segmentation (Red=Artery, Blue=Vein, Green=Junction)",
                    height=350,
                    elem_classes="output-section"
                )

        # ── Submit button (centred) ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("")
            with gr.Column(scale=2):
                analyze_btn = gr.Button(
                    "🚀 SUBMIT - Analyze Image",
                    variant="primary",
                    size="lg",
                    scale=2
                )
            with gr.Column(scale=1):
                gr.Markdown("")

        # ── Row 2: Artery + Vein masks ───────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>🔴 Artery Mask</b></div>')
                artery_output = gr.Image(
                    label="Extracted Artery Pixels",
                    height=350,
                    elem_classes="output-section"
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>🔵 Vein Mask</b></div>')
                vein_output = gr.Image(
                    label="Extracted Vein Pixels",
                    height=350,
                    elem_classes="output-section"
                )

        # ── Row 3: Detailed analysis panels ─────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>📊 Artery Analysis</b></div>')
                artery_viz_output = gr.Image(
                    label="Comprehensive Artery Analysis (Skeleton, Distance Map, Junctions, Segments)",
                    height=500,
                    elem_classes="analysis-section"
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>📊 Vein Analysis</b></div>')
                vein_viz_output = gr.Image(
                    label="Comprehensive Vein Analysis (Skeleton, Distance Map, Junctions, Segments)",
                    height=500,
                    elem_classes="analysis-section"
                )

        # ── Row 4: Summary + JSON ────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>📋 Analysis Summary</b></div>')
                summary_output = gr.Textbox(
                    label="Statistical Summary",
                    lines=20,
                    max_lines=20,
                    elem_classes="summary-section",
                    interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header"><b>📄 JSON Output</b></div>')
                json_output = gr.Code(
                    label="Complete Analysis Data (JSON)",
                    language="json",
                    lines=20,
                    max_lines=20,
                    elem_classes="json-section",
                    interactive=False
                )

        # ── Tips ─────────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("""
            ### 💡 Usage Tips
            - **Upload** clear fundus images (JPG, PNG) for best results
            - **Red** pixels = Arteries | **Blue** pixels = Veins | **Green** pixels = Junction points
            - **Analysis** extracts diameter, tortuosity, bifurcation angles, and more
            - **JSON data** can be downloaded for further processing or integration
            - All measurements are in **pixels** (convert to real units using image calibration)
            """)

        # ── Wire up button ────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_fundus_image,
            inputs=[input_image],
            outputs=[
                segmentation_output,
                artery_output,
                vein_output,
                artery_viz_output,
                vein_viz_output,
                json_output,
                summary_output,
            ]
        )

        # ── Footer ────────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("""
            <div style="text-align:center;padding:20px;color:#666;
                        border-top:1px solid #ddd;margin-top:20px;">
            <p><b>Integrated Retinal Vessel Analyzer v1.0</b> | For Research Use Only</p>
            <p>Combines deep learning segmentation with comprehensive geometric
               and topological vessel analysis</p>
            </div>
            """)

    return demo


# ================= MAIN =================

if __name__ == "__main__":
    print("🚀 Starting Integrated Retinal Vessel Analyzer...")
    demo = create_gradio_interface()
    print("✅ Ready! Launching Gradio interface...")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css
    )
