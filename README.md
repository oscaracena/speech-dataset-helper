
This is a Python/Gtk application that helps you in the task of recording and
annotating audio clips to generate a speech dataset used to train TTS models.

It uses the same format as LJ-Speech dataset, from Keith Ito and Linda Johnson.
It also trims and denoises the clips, auto-normalize labels and some other
minor tasks.

Btw, this is provided as-is. I made it for fun, and also to help me in other
projects. You can use it, if you want. This code is under GPLv3.

To run this application, first install its depends (listed on DEPENDS file).
On Debian and derived, you can run:

    make install-depends

Then, to launch, just run:

    ./sdh.py

Here is a screenshot:

![SDH Screenshot](images/screenshot.png)


# References

* https://arxiv.org/abs/1809.10460v1
* https://www.deepmind.com/publications/wavenet-a-generative-model-for-raw-audio
* https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio
* https://paperswithcode.com/paper/wavenet-a-generative-model-for-raw-audio#code

* TTS:
    * https://github.com/espnet/espnet
    * https://espnet.github.io/espnet/notebook/tts_cli.html
    * https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/
    * https://r9y9.github.io/wavenet_vocoder/
    * https://keithito.com/LJ-Speech-Dataset/

    * https://github.com/Kyubyong/tacotron
    * https://github.com/kyubyong/dc_tts
    * https://github.com/barronalex/Tacotron
    * https://github.com/begeekmyfriend/tacotron/tree/master
    * https://github.com/keithito/tacotron
    * https://github.com/Rayhane-mamah/Tacotron-2

    * https://github.com/NVIDIA/NeMo
    * https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/intro.html
    * https://github.com/NVIDIA/NeMo/blob/stable/tutorials/tts/NeMo_TTS_Primer.ipynb

* Sound Analysis / Synthesis
    * https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

* https://www.altexsoft.com/blog/audio-analysis/


# RECIPES

* CLI Denoise: https://unix.stackexchange.com/a/427343/229516
    - cut a noise sample with audacity
    - sox noise.wav -n noiseprof noise_profile
    - sox source.mp3 output.mp3 noisered noise_profile 0.31

* CLI Trim silence: https://stackoverflow.com/a/72426502/870503
    - sox source.wav output.wav silence 1 0.1 0.8% reverse silence 1 0.1 0.8% reverse
