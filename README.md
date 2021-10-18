# TensorRT PTQ hands on

You can run the `mnist_int8.ipynb` to finish this hands-on test. 
Before opening the notebook, you need to start a docker and do some port mapping as following.

1. Run `docker run --rm -it --gpus '"device=0"' -v /your_path:/workspace/hostdir -p xxxx:8888 nvcr.io/nvidia/tensorrt:21.09-py3`. `/your_path` is the dir you want to mount to docker, `xxxx` is the server port you will use to visit the notebook, and `8888` is the jupyter notebook port within the docker. 
2. In docker, please run `pip install jupyter && jupyter notebook --ip=0.0.0.0 --NotebookApp.token='' --no-browser --port=8888 --allow-root`
3. Tap Win+R and run `ssh -N -f -L localhost:xxxx:localhost:xxxx -p 22 user_ID@x.x.x.x`. `xxxx` is the port you used in step 1, `user_ID` is you ID to access to the server, and `x.x.x.x`is the server IP address.
4. In your host browser, please input `http://localhost:8889/`

Now you can open the .ipynb file. Please run every cell of this notebook.
Then, please finish the questions and TODO tasks in this notebook.
