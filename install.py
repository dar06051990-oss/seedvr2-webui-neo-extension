import launch

if not launch.is_installed("rotary-embedding-torch"):
    launch.run_pip("install rotary-embedding-torch")
