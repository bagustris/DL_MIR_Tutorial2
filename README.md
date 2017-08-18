# Tutorial: Deep Learning on Music Information Retrieval

(c) 2017 by Thomas Lidy, TU Wien - http://ifs.tuwien.ac.at/~lidy

Tutorial ini merupakan demo implementasi deep learning untuk permasalahan mir: music information retrieval. Anda yang ingin belajar deep learning atau ataupun music information retrieval dapat menggunakanannya.

Untuk tutorial ini, kita menggunakan iPython / Jupyter notebook yang mana kita bisa memprogram dan mengeksekusi skrip python secara interaktif menggunakan web browser sebagai IDE.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' content in your browser, by clicking on
the tutorial's filenames listed below in the GIT file listing (above, resp. on https://github.com/tuwien-musicir/DL_MIR_Tutorial ).

The tutorial will open in your browser for viewing.

### Interactive Coding

If you want to follow the Tutorials by actually executing the code on your computer, please [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the `DL_MIR_Tutorial` folder and start from the command line:

`ipython notebook` or `jupyter notebook`

Your web browser will open showing a list of files. Start the tutorials one after another by clicking on the following:

<b>Music_genre_classification.ipynb</b><br/>
   This tutorial shows how music is categorized into 1 of 10 music genres using the GTZAN music collection (see below).
   It includes audio and data preprocessing for Deep Learning and creating and training different architectures and parameters of a Convolutional Neural Network. It also includes techniques such as Batch Normalization, ReLU Activation and Dropout.


# Installation of Pre-requisites

## Install Python 2.7

Note: On most Mac and Linux systems Python is already pre-installed. Check with `python --version` on the command line whether you have Python 2.7.x installed.

Otherwise install Python 2.7 from https://www.python.org/download/releases/2.7/

## Install Python libraries:

### Mac, Linux or Windows

(on Windows leave out `sudo`)

```
sudo pip install ipython
```

Try if you can open 
```
ipython notebook
```
on the command line. Otherwise try to install:
```
sudo pip install jupyter
```

Then download or clone the Tutorials from this GIT repository:

```
git clone https://github.com/tuwien-musicir/DL_MIR_Tutorial.git
```
or download https://github.com/tuwien-musicir/DL_MIR_Tutorial/archive/master.zip <br/>
unzip it and rename the folder to `DL_MIR_Tutorial`.

Install the remaining Python libraries needed:

Either by:

```
sudo pip install Keras==1.2.1 Theano==0.8.2 scikit-learn>=0.17 pandas librosa
```

or, if you downloaded or cloned this repository, by:

```
cd DL_MIR_Tutorial
sudo pip install -r requirements.txt
```

### Install MP3 Decoder

If you want to use audio formats other than .wav files (e.g. .mp3, .flac, .au, .mp4), you have to install FFMPEG on you computer:

- Linux: install `ffmpeg`, via `sudo apt-get install ffmpeg`)
  - Untuk Ubuntu 14.04 dan 16.04, baca link berikut: https://askubuntu.com/questions/699502/ffmpeg-command-not-found
- Mac: download FFMPeg for Mac: http://ffmpegmac.net or if you use brew, execute: `brew install ffmpeg`
- Windows: download FFMpeg.exe from https://github.com/tuwien-musicir/rp_extract/blob/master/bin/external/win/ffmpeg.exe

Make sure that the exectuable is in a PATH found by the system.

## Configure Keras to use Theano

Since we use Theano as the Deep Learning computation backend, but Keras is configured to use TensorFlow by default, we have to change this in the `keras.json` configuration file, which is in the `.keras` folder of the user's HOME directory.

Copy the `keras.json` included in the `DL_MIR_Tutorial` to one of the following target directories (you can overwrite an existing file):

* Windows: `C:\Users\<user>\.keras\`
* Mac: `/Users/<user>/.keras`
* Linux: `/home/<user>/.keras`

An alternantive is to change these 2 lines in your `keras.json` file to the following:
```
{
    "image_dim_ordering": "th",
    "backend": "theano"
}
```

See https://keras.io/backend/ for details or http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/ for a step by step guide.

### Optional for GPU computation

If you want to train your neural networks on your GPU, also install the following (not needed for the tutorials):

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (optional, for further speedup)

To permanently configure Keras/Theano to use the GPU place a file `.theanorc` in your home directory with the following content:

```
[global]
device = gpu
floatX = float32
mode=FAST_RUN
```

### Check if installed correctly

To check whether Python, Keras and Theano were installed correctly, do:

`
python test_keras.py
`

If everything is installed correctly, it should print `Using Theano backend.`<br/>
If the GPU is configured correctly, it should also print `Using gpu device 0: GeForce GTX 980 Ti` or similar.


# Source Credits

## Python libraries

The following helper Python libraries are used in these tutorials:

* `audiofile_read.py` and `rp_extract.py`: by Thomas Lidy and Alexander Schindler, taken from the [RP_extract](https://github.com/tuwien-musicir/rp_extract) git repository
* `wavio.py`: by Warren Weckesser

## Data Sources

The data sets we use in the tutorials are from the following sources:

* GTZAN music genre data set:
by George Tzanetakis
1000 audio files with 30 sec. each, across 10 music genres, 100 audio files each

* GTZAN music speech data set: (currently not used)
by George Tzanetakis
Collected for the purposes of music/speech discrimination. 128 tracks, each 30 seconds long. Each class (music or speech) has 64 examples in 22050Hz Mono 16-bit WAV audio format.

both data sets available from:
http://marsyasweb.appspot.com/download/data_sets/
