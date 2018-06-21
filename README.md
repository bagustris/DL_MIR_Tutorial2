# Tutorial: Deep Learning on Music Information Retrieval

(c) 2017 by Thomas Lidy, TU Wien - http://ifs.tuwien.ac.at/~lidy

[bagustris](https://github.com/bagustris) forked this repo from http://github.com/tuwien-musicir/DL_MIR_Tutorial and updated freely in Indonesian language. Based on Indonesian School of Music Information Retrieval, August 2017.


---
Tutorial ini merupakan demo implementasi deep learning untuk permasalahan mir: music information retrieval. Anda yang ingin belajar deep learning ataupun music information retrieval dapat menggunakanannya.

Untuk tutorial ini, kita menggunakan iPython / Jupyter notebook di mana kita bisa memprogram dan mengeksekusi skrip python secara interaktif menggunakan web browser layaknya IDE (integrated development editor).

### Viewing Only
Jika anda hanya ingin melihat saja (artinya anda tidak (ingin) mengeksekusi skrip python didalam jupyter notebook, anda dapat membuka file berikut, https://github.com/bagustris/DL_MIR_Tutorial2/blob/master/Music_genre_classification.ipynb.

### Interactive Coding

Jika anda ingin mengikuti tutorial ini secara komprenhensif (dengan kata lain: anda ingin belajar sungguh-sungguh), anda harus menginstall program berikut dengan versi yang sama persis dengan yang dibutuhkan. Menginstall versi dari librari yang lebih tinggi atau lebih rendah menyebabkan program gagal berjalan. Tujuan dari tutorial ini adalah untuk membuktikan bahwa untuk memahami cara kerja deeplearning dan implementasinya untuk klasifikasi genre musik. Jadi, pada step awal tutorial ini, kita harus memastikan bahwa program yang dirancang **just works**. Selanjutnya anda bisa memodifikasi sendiri bila telah berhasil menjalankan program deeplearning pada tutorial ini.

### Step by step
Langkah-langkah menjalakan program deeplearning python (keras berbasis theano) pada tutorial ini:
1. Clone repository ini (atau dowload zip-nya),
`git clone https://github.com/bagustris/DL_MIR_Tutorial2.git`
2. Install [program/library/module yang dibutuhkan](#installation-of-pre-requisites)
3. Pindah direktori pada DL_MIR_Tutorial2 (hasil clone)
3. Jalankan ipython atau jupyter notebook sbb:
`ipython notebook` or `jupyter notebook`
4. Buka file Music_genre_classification2.ipynb
5. Jalankan tiap baris file Music_genre_classification2.ipynb tersebut hingga hasilnya sama atau mirip dengan file Music_genre_classification.ipynb


### Music_genre_classification2.ipynb 
Ini adalah file utama kita. Jalankan tiap baris skrip python pada file tersebut (tekan Ctrl+Enter) dan lihat hasilnya. Jika tidak ada error, lanjutkan pada baris selanjutnya, jika ada error, cari errornya dimana dan perbaiki, kemudian jalankan lagi.

Output yang diharapkan dari file tersebut adalah bagaimana mengklasifikasikan genre musik dengan deep learning berbasis python keras dan theano menggunakan teknik *Convolutional Neural Network*. Teknik deep learning yang diimplementasikan pada skrip tersebut juga mencangkup teknik *Batch Normalization, ReLU Activation dan Dropout*.


## Installation of Pre-requisites

### Install Python 2.7
Kita menggunakan python versi 2.7, bukan versi 3(.5). Pada kebanyakan sistem Unix, yakni Linux atau Mac, python 2.7 sudah terinstall *by default*.  Silahkan cek versi python anda dengan menjalankan perintah `python --version` pada terminal Linux.

Sangat disarankan menggunakan Ubuntu 16.04 dimana tutorial ini dibuat.

Jika python 2.7 belum terinstall pada sistem anda, silahkan merefer kesini:
https://www.python.org/download/releases/2.7/

### Install Python libraries:

#### Mac, Linux or Windows

(pada windows tanpa memakai `sudo`)

```
sudo -H pip install jupyter
```

Kemudian ikuti langkah pertama [di atas](#step-by-step) atau download dan ekstrak dari link berikut: https://github.com/tuwien-musicir/DL_MIR_Tutorial/archive/master.zip <br/>

Ganti nama hasil extract tadi dengan nama baru: `DL_MIR_Tutorial2`.

Install library python yang dibutuhkan dengan **SALAH SATU** cara berikut.

Cara pertama:

```
sudo -H pip install Keras==1.2.1 Theano==0.8.2 scikit-learn>=0.17 pandas librosa
```

Cara kedua:
Dari dalam folder `DL_MIR_Tutorial2`:
```
cd DL_MIR_Tutorial2
sudo -H pip install -r requirements.txt
```
Perhatikan versi library yang diinstall untuk tutorial ini, kesalahan versi saat menginstall library akan menyebabkan program pada tutorial ini tidak bisa dijalankan.

#### Dengan virtualenv
Virtual env memudahkan kita untuk menginstall paket python dengan mengisolasi environment python secara custom. Jadi instalasi python utama dan yang lain tidak terganggu. Baca tutorialnya [di sini](http://www.bagustris.tk/2017/11/tutorial-python-virtualenv.html). Dengan virtualenv, langkah-langkah di atas dapat disederhanakan menjadi berikut,

    $ virtualenv pymir
    $ source pymir/bin/activate
    $ pip2 install -r requirement.txt
    $ ipython kernel install --user -name=pymir #install ipykernel jika belum ada
    $ jupyter-notebook

### Install MP3 Decoder

Karena alasan bandwidth, kita menggunakan database file musik dengan format .mp3 (umumnya format .wav yang banyak dipakai untuk proses manipulasi audio). Jadi kita harus mempunyai codec untuk mengkonversi .mp3 menjadi .wav secara **on the fly**.

- Linux: install `ffmpeg`, via `sudo apt-get install ffmpeg`)
  - Untuk Ubuntu 14.04 dan 16.04, baca link berikut: https://askubuntu.com/questions/699502/ffmpeg-command-not-found
- Mac: download FFMPeg Untuk Mac: http://ffmpegmac.net jika anda menggunakan `brew` jakankan: `brew install ffmpeg`
- Windows: download FFMpeg.exe dari https://github.com/tuwien-musicir/rp_extract/blob/master/bin/external/win/ffmpeg.exe

Pastikan file executable `ffmpeg` berada pada lingkup pencarian system. Secara otomatis, pada sistem Ubuntu/Linux berada pada `/usr/bin`.

### Configure Keras to use Theano

Karena kita menggunakan Theano sebagai backend komputasi Deep Learning, namun Keras dikonfigurasi untuk menggunakan Tensor Flow *by default*, maka kita perlu mengkonfigurasi keras agar menggunakan backend Theano. Caranya adalah dengan merubah isi file `.keras/keras.json` pada HOME direktori anda (pada sistem Linux). Copy file keras.json pada direktori `DL_MIR_Tutorial2` ini pada HOME direktori anda untuk meng-overwrite file yang telah ada.

* Windows: `C:\Users\<user>\.keras\`
* Mac: `/Users/<user>/.keras`
* Linux: `/home/<user>/.keras`

Sebagai alternatif, anda dapat merubah isi file `.keras/keras.json` pada HOME direktori anda dengan isi berikut.
```
{
    "image_dim_ordering": "th",
    "backend": "theano"
}
```

Lihat https://keras.io/backend/ untuk lebih detail, atau buka http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/ untuk langkah langkah panduan di Windows 10 (tidak disarankan).

### Optional for GPU computation

Jika anda ingin menggunakan GPU (graphical processing unit) untuk komputasi Deep Learning, anda harus menginstall driver GPU. Lihat beberapa link berikut.

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (optional, for further speedup)

Konfigurasi agar Keras/Theano adalah dengan membuat file `.theanorc` dalam HOME direktori dengan isi sebagai berikut:

```
[global]
device = gpu
floatX = float32
mode=FAST_RUN
```

### Check if installed correctly

Untuk mengecek apakan keras telah terkonfigurasi dengan theano jalankan perintah berikut pada terminal
`
python test_keras.py
`

Atau ketik `python` pada terminal dan isikan `import keras` pada konsol python.

Jika muncul `Using Theano backend`, artinya keras telah konfigurasi. Jika muncul error, carilah erronya dimana.
Jika dikonfigurasi menggunakan GPU maka akan muncul `Using gpu device 0: GeForce GTX 980 Ti` atau sejenisnya.


## Source Credits

### Python libraries

Beberapa library python yang digunakan pada tutorial ini.

* `audiofile_read.py` and `rp_extract.py`: by Thomas Lidy and Alexander Schindler, diambil dari repo berikut: [RP_extract](https://github.com/tuwien-musicir/rp_extract).
* `wavio.py`: by Warren Weckesser

### Data Sources

Dataset (file mp3) diambil dari link paling bawah halaman ini. Ekstrak data file tersebut dan letakkan pada folder `data` yang letaknya **sejajar** dengan folder `DL_MIR_Tutorial2`. Selamat mencoba, jika menemui kesulitan [silahkan bertanya](https://github.com/bagustris/DL_MIR_Tutorial2/issues).

* GTZAN music genre data set:
by George Tzanetakis
1000 audio files with 30 sec. each, across 10 music genres, 100 audio files each

* GTZAN music speech data set: (currently not used)
by George Tzanetakis
Collected for the purposes of music/speech discrimination. 128 tracks, each 30 seconds long. Each class (music or speech) has 64 examples in 22050Hz Mono 16-bit WAV audio format.

both data sets available from:
http://marsyasweb.appspot.com/download/data_sets/
