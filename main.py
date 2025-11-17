import gradio as gr
import subprocess
import tempfile
import os
import shutil

target_script = "people_counter.py"
if not os.path.exists(target_script):
    raise FileNotFoundError(f"Backend script '{target_script}' not found! Please create people_counter.py")

def run_counter(video_file):
    if video_file is None:
        return "ERROR: Pehle video upload karo!", None

    try:
        if isinstance(video_file, str):
            # Gradio gives file path
            temp_input = os.path.join(os.getcwd(), "uploaded_video.mp4")
            shutil.copyfile(video_file, temp_input)
        else:
            # File-like object
            temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_obj.write(video_file.read())
            temp_obj.close()
            temp_input = temp_obj.name
    except Exception as e:
        return f"Error saving uploaded video: {e}", None

    temp_output = os.path.join(os.getcwd(), "output_with_tracking.mp4")

    if os.path.exists(temp_output):
        try:
            os.remove(temp_output)
        except:
            pass

    cmd = ["python", target_script, temp_input, temp_output]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return f"Error running backend: {e}", None

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if os.path.exists(temp_output):
        return log, temp_output
    else:
        return log + "\nOutput file not found!", None

custom_css = """
#app-title {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: white;
    padding: 12px;
    background: linear-gradient(90deg, #1d2671, #c33764);
    border-radius: 12px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(css=custom_css) as ui:
    gr.HTML("<div id='app-title'>People Counter — YOLOv8 + IoU Tracking</div>")
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Upload CCTV / Phone Video")
            run_btn = gr.Button("Process Video")

        with gr.Column(scale=1):
            output_box = gr.Textbox(label="Logs", lines=15)
            download_out = gr.File(label="Download Processed Video")

    run_btn.click(run_counter, inputs=video_in, outputs=[output_box, download_out])

if __name__ == "__main__":
    ui.launch()
