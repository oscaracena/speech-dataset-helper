#!/usr/bin/env python3

import os
import json
import logging
import time
import re
from pathlib import Path
from argparse import ArgumentParser
from threading import Thread, Event
from queue import Queue
from functools import partial
import tempfile
import gi
import wave
import nltk
import sox
import pyaudio
from num2words import num2words

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib

logging.basicConfig(level=logging.INFO)
logging.getLogger("sox").setLevel(logging.ERROR)
log = logging.getLogger("SDH")


def SoxTransformer():
    tfm = sox.Transformer()
    tfm.set_globals(
        verbosity = 0
    )
    return tfm


class Recorder(Thread):
    def __init__(self, dev):
        super().__init__()
        self._audio = dev
        self._buffer_size = 1024
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 22050
        self._frames = []
        self._stream = None
        self._cmd_q = Queue()

        self.recording = False
        self.daemon = True
        self.start()

    def run(self):
        while True:
            while not self._cmd_q.empty():
                cmd = self._cmd_q.get()
                cmd()

            if self.recording:
                data = self._stream.read(self._buffer_size)
                self._frames.append(data)
            else:
                time.sleep(0.1)

    def clear(self):
        self._cmd_q.put(self._do_clear)

    def rec_start(self):
        self._cmd_q.put(self._do_rec_start)

    def rec_stop(self):
        self._cmd_q.put(self._do_rec_stop)

    def rec_save(self, path, block=False):
        if not block:
            self._cmd_q.put(partial(self._do_rec_save, path))
        else:
            ev = Event()
            self._cmd_q.put(partial(self._do_rec_save, path, ev))
            assert ev.wait(10)

    def _do_clear(self):
        self._frames = []

    def _do_rec_start(self):
        self._stream = self._audio.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            output=False,
            frames_per_buffer=self._buffer_size)
        self.recording = True

    def _do_rec_stop(self):
        self.recording = False
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

    def _do_rec_save(self, path, ev=None):
        wf = wave.open(path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(
            self._audio.get_sample_size(self._format))
        wf.setframerate(self._rate)
        wf.writeframes(b"".join(self._frames))
        wf.close()
        if ev:
            ev.set()


class Player(Thread):
    def __init__(self, dev):
        super().__init__()
        self._audio = dev
        self._stream = None
        self._wavef = None
        self._cmd_q = Queue()

        self.playing = False
        self.daemon = True
        self.start()

    def run(self):
        while True:
            while not self._cmd_q.empty():
                cmd = self._cmd_q.get()
                cmd()

            if self.playing and not self._stream.is_active():
                log.info(" clip ended")
                self._do_audio_stop()

            time.sleep(0.1)

    def audio_stop(self):
        self._cmd_q.put(self._do_audio_stop)

    def audio_play(self, path):
        self._cmd_q.put(partial(self._do_audio_play, path))

    def _do_audio_stop(self):
        if not self.playing:
            return

        self.playing = False
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None
        self._wavef.close()
        self._wavef = None

    def _do_audio_play(self, path):
        self._wavef = wave.open(path, 'rb')

        def get_chunk(in_data, frame_count, time_info, status):
            chunk = self._wavef.readframes(frame_count)
            return (chunk, pyaudio.paContinue)

        self._stream = self._audio.open(
            format=self._audio.get_format_from_width(
                self._wavef.getsampwidth()),
            channels=self._wavef.getnchannels(),
            rate=self._wavef.getframerate(),
            output=True,
            stream_callback=get_chunk,
        )

        self.playing = True


class AudioManager:
    def __init__(self, dset):
        self._audio_dev = pyaudio.PyAudio()

        self._clips_dir = dset.clips_dir
        self._noise_prof = dset.noise_profile

        self._counter = self._count_clips()
        self._recorder = Recorder(self._audio_dev)
        self._player = Player(self._audio_dev)
        self._sentence = None

    def __del__(self):
        if self._audio_dev is not None:
            self._audio_dev.terminate()

    def is_recording(self):
        return self._recorder.recording

    def is_playing(self):
        return self._player.playing

    def stop_clip(self):
        log.info(" stop playing!")
        self._player.audio_stop()

    def stop_recording(self):
        log.info(" stop recording!")
        self._recorder.rec_stop()

    def start_recording(self, sentence):
        log.info(" start recording...")
        self._sentence = sentence
        self._recorder.clear()
        self._recorder.rec_start()

    def save_clip(self):
        path = self._sentence.get("clip-path")
        if path is None:
            path = self._get_next_clip_file()
            self._sentence["clip-path"] = path

        if not self._clips_dir.exists():
            self._clips_dir.mkdir()

        log.info(f" saving clip as '{path}'...")
        self._recorder.rec_save(
            (self._clips_dir / path).as_posix(), block=True)

    def filter_clip(self):
        src_path = self._sentence.get("clip-path")
        if src_path is not None:
            src_path = self._clips_dir / src_path
        if src_path is None or not src_path.exists():
            log.error(f" could not run filter, clip not saved")
            return

        log.info(f" denoising and trimming clip...")

        # sox needs an input file to be different than output
        # I need to create a copy here
        temp_wav = tempfile.mkstemp()[1]
        with src_path.open("rb") as src:
            with open(temp_wav, "wb") as dst:
                dst.write(src.read())
        src_path.unlink()

        tfm = SoxTransformer()

        if self._noise_prof.exists():
            tfm.noisered(self._noise_prof.as_posix(), 0.25)
        else:
            log.warning(" not denoising file, noise profile not generated!")

        tfm.silence(location=1, silence_threshold=0.5,
            buffer_around_silence=True)
        tfm.silence(location=-1, silence_threshold=0.5,
            buffer_around_silence=True)
        tfm.build(temp_wav, src_path.as_posix())
        Path(temp_wav).unlink()

    def play_clip(self, sentence):
        path = sentence.get("clip-path")
        if path is None:
            log.warning(" this sentence does not have a recording yet!")
            return

        full_path = self._clips_dir / path
        if not full_path.exists():
            log.warning(f" '{path}' is missing!")
            return

        log.info(f" playing clip: {path}")
        self._player.audio_play(full_path.as_posix())

    def update_noise_profile(self):
        """This will record 2 seconds of audio, then strip 0.5 at each
        end, and save it as a sox noise profile."""
        if self._recorder.recording:
            log.warning(" there is a recoding in progress, can't "
                "get noise profile.")
            return

        if self._player.playing:
            self.stop_clip()

        # FIXME: this will freeze the user interface while it is
        # working!! Fix it!

        log.info(" acquiring background noise...")
        noise_wav = (self._clips_dir / "noise-profile.wav") \
            .as_posix()
        self._recorder.rec_start()
        time.sleep(2)
        self._recorder.rec_stop()
        temp_wav = tempfile.mkstemp()[1]
        self._recorder.rec_save(temp_wav, block=True)

        log.info(" converting noise to sox profile...")

        if self._noise_prof.exists():
            self._noise_prof.unlink()

        tfm = SoxTransformer()
        tfm.trim(0.5, 1.5)
        tfm.build(temp_wav, noise_wav)
        tfm.noiseprof(noise_wav, self._noise_prof.as_posix())
        Path(temp_wav).unlink()
        log.info(f" noise profile saved in '{self._noise_prof.name}'")

    def _get_next_clip_file(self):
        self._counter += 1
        while True:
            file = f"clip-{self._counter:06d}.wav"
            if (self._clips_dir / file).exists():
                self._counter += 1
                continue
            return file

    def _count_clips(self):
        if not self._clips_dir.exists():
            return 0
        return len(list(self._clips_dir.glob("clip-*.wav")))


class DatasetProject:
    def __init__(self, path, data=None):
        self._prj_path = path
        data = data or {}

        self._current = data.get("current", 0)
        self._text = data.get("text", [])
        self._clips_dir = data.get("clips-dir", "wav")
        self._meta_file = data.get("metadata", "transcripts.csv")
        self._noise_prof_file = data.get(
            "noise-profile", "noise_profile.sox")

        # public static properties
        self.project_name = data.get("name", "unknown")

    @classmethod
    def load(cls, path):
        data = json.load(open(path, "r"))
        log.info(f" project succesfully loaded from '{path}'")
        return cls(path, data)

    def save(self):
        self._save_project()
        self._save_metadata()

    def _save_project(self):
        data = {
            "name": self.project_name,
            "current": self._current,
            "text": self._text,
            "clips-dir": self._clips_dir,
            "metadata": self._meta_file,
            "noise-profile": self._noise_prof_file,
        }
        json.dump(data, open(self._prj_path, "w"))
        log.info(f" project succesfully saved to '{self._prj_path}'")

    def _save_metadata(self):
        cdir = self.clips_dir
        mfile = self.project_dir / self._meta_file
        with mfile.open("w") as dst:
            for s in self._text:
                clip = s.get("clip-path")
                if clip is None or not (cdir / clip).exists():
                    continue
                line = self._sentence_to_meta(s)
                dst.write(line)
        log.info(f" metadata saved to '{self._meta_file}'")

    def _sentence_to_meta(self, s):
        normal = s["content"].strip()

        # NOTE: Is very difficult to constrain here all the possible
        # options. I'll do the most frecuent changes, and the user
        # is responible for fixing the rest.

        # roman numbers to cardinal (or ordinal, if are for centuries <= 10)
        rex_cent = re.compile(
            r'(\b[MCDXLVI]{1,6}\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})')
        fields = rex_cent.split(normal)
        ordinal = None
        for i, w in enumerate(fields[:]):
            if rex_cent.match(w):
                number = self._roman_to_number(w.lower())
                numtype = "cardinal"
                if ordinal and number < 11:
                    numtype = "ordinal"
                fields[i] = num2words(number, lang="es", to=numtype)
            ordinal = any(w.strip().endswith(x) for x in ["siglo", "siglos"])
        normal = "".join(fields)

        # numbers into words
        fields = re.split(r'(\d+)', normal)
        for i, w in enumerate(fields[:]):
            if w is None:
                continue
            if w.isnumeric():
                fields[i] = num2words(int(w), lang="es")
        normal = "".join(fields)

        return f"{s['clip-path']}|{s['content']}|{normal.lower()}\n"

    def _roman_to_number(self, s):
        total = 0
        romans = {
            'i': 1, 'v': 5, 'x': 10, 'l': 50,
            'c': 100, 'd': 500, 'm': 1000,
        }

        try:
            s = s.lower()
            for i in range(len(s) - 1):
                if romans[s[i + 1]]  > romans[s[i]]:
                    total -= romans[s[i]]
                else:
                    total += romans[s[i]]

            total += romans[s[-1]]
        except Exception as err:
            log.error(" invalid roman number!!")
            return None

        return total

    @property
    def size(self):
        return len(self._text)

    @property
    def current(self):
        return self._text[self._current]

    @property
    def project_dir(self):
        return Path(self._prj_path).parent.absolute()

    @property
    def clips_dir(self):
        return (Path(self._prj_path).parent / self._clips_dir) \
            .absolute()

    @property
    def noise_profile(self):
        return (Path(self._prj_path).parent / self._noise_prof_file) \
            .absolute()

    def _new_sentence(self, content=""):
        content = content.strip()
        # print("    >>", len(content), content)
        return {
            "content": content,
            "hidden": False,
            "clip-path": None,
        }

    def _get_sentence(self, buff, slen=100, span=20):
        """If buffer is enough, extract a sentence and return
        both, sentence and remainder."""

        try:
            # Try to get sentences using nltk
            tokens = nltk.tokenize.sent_tokenize(buff)

            # Then, get the first one. If its size is in range
            # [slen-span,slen+span], then return it.
            # If is shorter, then try adding the next sentence
            s = ""
            for i, t in enumerate(tokens):
                s += f" {t}"
                if (slen - span) < len(s) < (slen + span):
                    return s, " ".join(tokens[i + 1:])

        except Exception as err:
            log.warning(f" nltk did not work: {err}")

        # In any case, if sentence is too big, then try
        # to split using colons or other mid-sentnce joints.
        for t in [",", ":"]:
            i = buff.find(t, slen - span, slen + span)
            if i != -1:
                return buff[:i + len(t)], buff[i + len(t):]

        # Or just using some kind of white space
        for t in [" ", "\t"]:
            i = buff.find(" ", slen - 8, slen + 8)
            if i != -1:
                return buff[:i + len(t)], buff[i + len(t):]

        # last restort
        return buff[:slen], buff[slen:]

    def import_text(self, path):
        self._text = []
        self._current = 0

        slen = 100
        span = 20

        # keep sentences between 100 and 120 chars, so they could
        # be read in under 10 seconds
        buff = ""
        with open(path) as src:
            for line in src.readlines():
                line = line.strip()
                # print("    ..", line)

                # if line is empty, force to save buffer as is
                if not line:
                    if buff:
                        self._text.append(self._new_sentence(buff))
                        buff = ""
                    continue

                buff += f" {line}"

                # if we can extract a new sentence, then do it
                while len(buff) > slen + span:
                    # print("        in:", len(buff))
                    s, buff = self._get_sentence(buff, slen, span)
                    # print("        out:", len(s), len(buff))
                    self._text.append(self._new_sentence(s))

        # after all, if there is still something, save it
        if buff:
            self._text.append(self._new_sentence(buff))

    def next(self, show_hidden=False):
        if self._current + 1 >= len(self._text):
            return

        self._current += 1
        if not show_hidden:
            if self._text[self._current].get("hidden", False):
                self.next(show_hidden)

    def prev(self, show_hidden=False):
        if self._current <= 0:
            return

        self._current -= 1
        if not show_hidden:
            if self._text[self._current].get("hidden", False):
                self.prev(show_hidden)

    def context(self, show_hidden=False):
        prev, curr, next = None, None, None
        if self._current > 0:
            prev = self._text[self._current - 1]
        if self._current >= 0 and self._current < len(self._text):
            curr = self._text[self._current]
        if self._current + 1 < len(self._text):
            next = self._text[self._current + 1]
        return prev, curr, next

    def delete_current(self):
        if self._current >= 0 and self._current < len(self._text):
            self._text.pop(self._current)
            if self._current >= len(self._text):
                self._current -= 1

    def hide_current(self):
        if self._current < len(self._text):
            curr = self._text[self._current]
            curr["hidden"] = not curr.get("hidden", False)

    def add_word(self):
        if self._current + 1 >= len(self._text):
            log.warning("can not add a new word, this is the last sentence.")
            return

        next_sentence = self._text[self._current + 1]["content"]
        words = next_sentence.split(" ")
        w = words.pop(0)
        self._text[self._current + 1]["content"] = " ".join(words)
        if w:
            self._text[self._current]["content"] += f" {w}"

    def sub_word(self):
        if self._current + 1 >= len(self._text):
            self._text.append(self._new_sentence())

        curr_sentence = self._text[self._current]["content"]
        words = curr_sentence.split(" ")
        w = words.pop(-1)
        self._text[self._current]["content"] = " ".join(words)
        if w:
            next_sentence = f"{w} " + self._text[self._current + 1]["content"]
            self._text[self._current + 1]["content"] = next_sentence


class TimerCtrl:
    def __init__(self, level, label):
        self._lvl = level
        self._label = label

        self._time = 0
        self._running = None
        self._update()

    def _update(self):
        self._label.set_markup(f"{self._time:.01f} s")
        self._lvl.set_value(
            min(self._lvl.get_max_value(), self._time))

    def _on_tick(self):
        self._time += 0.1
        self._update()
        return True

    def is_running(self):
        return self._running != None

    def stop(self):
        if self._running is None:
            return
        GLib.source_remove(self._running)
        self._running = None
        self._update()

    def reset(self):
        self._time = 0
        self.stop()

    def start(self):
        self._running = GLib.timeout_add(100, self._on_tick)
        self._update()


@Gtk.Template(filename="ui.glade")
class GUIWindow(Gtk.Window):
    __gtype_name__ = "mainWin"

    file_dlg = Gtk.Template.Child("fileDialog")
    pre_text_lbl = Gtk.Template.Child("preText")
    to_read_lbl = Gtk.Template.Child("textToRead")
    post_text_lbl = Gtk.Template.Child("postText")
    status_lbl = Gtk.Template.Child("statusLabel")
    sentence_stat = Gtk.Template.Child("sentenceStatusIcon")
    prj_info = Gtk.Template.Child("projectInfoLbl")

    time_level = Gtk.Template.Child("timerLvl")
    time_lbl = Gtk.Template.Child("timerLbl")

    prj_filter = Gtk.Template.Child("projectFileFilter")
    text_filter = Gtk.Template.Child("textFileFilter")

    def __init__(self, args):
        super().__init__()

        self._show_hidden = False
        self._play_on_finish = False
        self._dset = None
        self._auman = None

        # setup file dialog
        self.file_dlg.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        # define and apply custom css styles
        with open("style.css", "br") as src:
            css = src.read()

        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(css)
        context = Gtk.StyleContext()
        screen = Gdk.Screen.get_default()
        context.add_provider_for_screen(
            screen, css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.time_level.add_offset_value(
            "low", 0.7 * self.time_level.get_max_value())
        self.time_level.add_offset_value(
            "high", 0.9 * self.time_level.get_max_value())
        self.time_level.add_offset_value(
            "full", self.time_level.get_max_value())

        # object to handle the time status
        self._timer = TimerCtrl(
            self.time_level, self.time_lbl)

        # create a text helper if file is provided
        if args.open_project:
            self._open_project(args.open_project)

    def _set_project_info(self,):
        info = self._dset.project_name
        self.prj_info.set_markup(f"Proyecto: <b>{info}</b>")

    def _open_project(self, path):
        self._dset = DatasetProject.load(path)
        self._auman = AudioManager(self._dset)

        self._set_project_info()
        if self._dset.size == 0:
            self._status("¡Proyecto vacío! Importa el cuerpo de "
                "texto para comenzar.")
            return
        self._update_text()

    def _open_dialog(self, filter, action):
        filter = {
            "prj": self.prj_filter,
            "txt": self.text_filter,
        }[filter]

        action = {
            "open": Gtk.FileChooserAction.OPEN,
            "save": Gtk.FileChooserAction.SAVE,
        }[action]

        self.file_dlg.set_action(action)
        for f in self.file_dlg.list_filters():
            self.file_dlg.remove_filter(f)
        self.file_dlg.set_filter(filter)
        self.file_dlg.set_current_folder(os.path.abspath("."))

        response = self.file_dlg.run()
        self.file_dlg.hide()
        if response != Gtk.ResponseType.OK:
            return None
        return self.file_dlg.get_filename()

    def _status(self, msg):
        self.status_lbl.set_markup(msg)
        self.pre_text_lbl.hide()
        self.to_read_lbl.hide()
        self.post_text_lbl.hide()
        self.status_lbl.show()

    def _update_icon(self, current):
        icon = "gtk-no"
        if current is not None:
            clip_path = current.get("clip-path")
            if clip_path and (self._dset.clips_dir / clip_path).exists():
                icon = "gtk-yes"
        self.sentence_stat.set_from_icon_name(icon, Gtk.IconSize.BUTTON)

    def _update_text(self):
        context = self._dset.context(self._show_hidden)
        labels = (self.pre_text_lbl, self.to_read_lbl, self.post_text_lbl)

        for s, w in zip(context, labels):
            text = ""
            if s is not None:
                text = s.get("content")
                style = w.get_style_context()
                if s.get("hidden", False):
                    style.add_class("hidden-sentence")
                else:
                    style.remove_class("hidden-sentence")
            w.set_markup(text)

        # set audio present icon
        self._update_icon(context[1])

        if any(context):
            self.status_lbl.hide()
            self.pre_text_lbl.show()
            self.to_read_lbl.show()
            self.post_text_lbl.show()

    @Gtk.Template.Callback()
    def on_destroy(self, *args):
        Gtk.main_quit()

    @Gtk.Template.Callback()
    def on_open_project(self, *args):
        path = self._open_dialog("prj", "open")
        if path is not None:
            self._open_project(path)

    @Gtk.Template.Callback()
    def on_new_project(self, *args):
        prjdir = self._open_dialog("prj", "save")
        if prjdir is None:
            return

        os.mkdir(prjdir)
        path = prjdir + "/project.sdhp"
        name = Path(prjdir).name
        self._dset = DatasetProject(path, {"name": name})
        self._dset.save()
        self._auman = AudioManager(self._dset)
        self._update_text()
        self._set_project_info()
        self._status("Proyecto creado. Importa el cuerpo de texto para comenzar.")

    @Gtk.Template.Callback()
    def on_save_project(self, *args):
        if not self._dset:
            self._status("No hay proyecto que guardar.")
            return
        self._dset.save()

    @Gtk.Template.Callback()
    def on_import_file(self, *args):
        if not self._dset:
            self._status("Crea o abre primero un proyecto para poder importar texto.")
            return

        path = self._open_dialog("txt", "open")
        if path is not None:
            self._dset.import_text(path)
            if self._dset.size > 0:
                self._update_text()

    @Gtk.Template.Callback()
    def on_next_part(self, *args):
        if self._dset is None:
            return
        if self._timer.is_running():
            self._timer.stop()
        self._dset.next(self._show_hidden)
        self._update_text()

    @Gtk.Template.Callback()
    def on_prev_part(self, *args):
        if self._dset is None:
            return
        if self._timer.is_running():
            self._timer.stop()
        self._dset.prev(self._show_hidden)
        self._update_text()

    @Gtk.Template.Callback()
    def on_delete_sentence(self, *args):
        if self._dset is None:
            return
        self._dset.delete_current()
        self._update_text()

    @Gtk.Template.Callback()
    def on_hide_sentence(self, *args):
        if self._dset is None:
            return
        self._dset.hide_current()
        self._update_text()

    @Gtk.Template.Callback()
    def on_show_hidden(self, menu):
        self._show_hidden = menu.get_active()
        self._update_text()

    @Gtk.Template.Callback()
    def on_play_on_finish(self, menu):
        self._play_on_finish = menu.get_active()

    @Gtk.Template.Callback()
    def on_sub_word(self, *args):
        if self._dset is None:
            return
        self._dset.sub_word()
        self._update_text()

    @Gtk.Template.Callback()
    def on_add_word(self, *args):
        if self._dset is None:
            return
        self._dset.add_word()
        self._update_text()

    @Gtk.Template.Callback()
    def on_test_time(self, *args):
        if self._auman.is_recording():
            log.warning(" esto no es posible mientras se graba.")
            return

        if self._timer.is_running():
            self._timer.stop()
        else:
            self._timer.reset()
            self._timer.start()

    @Gtk.Template.Callback()
    def on_next_and_test_time(self, *args):
        self.on_next_part()
        self.on_test_time()

    @Gtk.Template.Callback()
    def on_next_and_record(self, *args):
        if self._auman is None:
            log.warning(" no hay ningun proyecto activo, nada que hacer.")
            return

        # toggle old recording first
        if self._auman.is_recording():
            self.on_record()

        # start current recording
        self.on_next_part()
        self.on_record()

    @Gtk.Template.Callback()
    def on_delete_clip(self, *args):
        if not self._dset:
            log.warning(" no hay ningun proyecto activo, nada que hacer.")
            return

        filename = self._dset.current.get("clip-path")
        if filename:
            clip_path = self._dset.clips_dir / filename
            if clip_path.exists():
                clip_path.unlink()
                log.warning(f" clip borrado: {filename}")
            self._dset.current["clip-path"] = None
            self._update_icon(self._dset.current)

    @Gtk.Template.Callback()
    def on_next_and_play(self, *args):
        if self._auman is None:
            log.warning(" no hay ningun proyecto activo, nada que hacer.")
            return

        # if recording, toggle record to avoid loosing it
        if self._auman.is_recording():
            self.on_record()

        if self._auman.is_playing():
            self._auman.stop_clip()

        self.on_next_part()
        self.on_play()

    @Gtk.Template.Callback()
    def on_record(self, *args):
        if self._auman is None:
            log.warning(" no hay ningun proyecto activo, nada que grabar.")
            return

        if self._auman.is_playing():
            self._auman.stop_clip()

        if self._auman.is_recording():
            self._timer.stop()
            self._auman.stop_recording()
            self._auman.save_clip()
            self._update_icon(self._dset.current)

            # this needs to be done when clip is already saved
            self._auman.filter_clip()
            log.info(" clip ready!")

            if self._play_on_finish:
                self.on_play()

        else:
            self._auman.start_recording(self._dset.current)
            self._timer.reset()
            self._timer.start()

    @Gtk.Template.Callback()
    def on_play(self, *args):
        if self._auman is None:
            log.warning(" no hay ningun proyecto activo, nada que grabar.")
            return
        if self._auman.is_recording():
            log.warning(" no se puede reproducir mientras se graba!")
            return

        if self._auman.is_playing():
            self._auman.stop_clip()
        else:
            self._auman.play_clip(self._dset.current)

    @Gtk.Template.Callback()
    def on_get_noise_profile(self, *args):
        if self._auman is None:
            log.warning(" abre o crea un proyecto antes de esto.")
            return

        self._auman.update_noise_profile()


if __name__ == "__main__":
    nltk.download("punkt", os.path.abspath("."))

    parser = ArgumentParser()
    parser.add_argument("-o", "--open-project",
        help="open SDH project file")

    args = parser.parse_args()

    window = GUIWindow(args)
    window.show()
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass
