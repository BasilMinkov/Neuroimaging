import time

import numpy as np
import rtmidi
from tqdm import tqdm

from data.nfblab_data import NFBLabData
from generator import stream_generator_in_a_thread
from inlets.lsl_inlet import LSLInlet


class LMInterface:

    def __init__(self, source="generator"):

        self.source = source

        if self.source is "generator":
            stream_generator_in_a_thread('NVX136_Data')
            self.lsl = LSLInlet(name='NVX136_Data')
        else:
            self.lsl = LSLInlet(name=self.source)

        self.fq = round(self.lsl.get_frequency())
        self.sleep_time = 1 / self.lsl.get_frequency()
        self.running = True

        print("Fq: ", self.fq, ", Sleep time: ", self.sleep_time)

        path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/FB_pilot_feb2_11-01_13-36-17/"
        channels_path = "/Users/basilminkov/Scripts/python3/Neuroimaging/static/chanlocs_mod.mat"

        self.eeg_data = NFBLabData(path, channels_path, parse=True, get_info=False)

        self.baseline_time = 10
        self.baseline_values = []

        self.calculate_baseline()
        self.start_midi_stream()

    def calculate_baseline(self):

        print("Calculating baseline...")

        for _ in tqdm(range(self.baseline_time)):

            y = []

            for __ in range(self.fq):

                time.sleep(self.sleep_time)
                chunk = self.lsl.get_next_chunk()[0]  # self.nfblab_data.channels_list.index("P4")
                try:
                    value = chunk[0, self.eeg_data.channels_list.index("Fp1")]
                    y.append(value)
                except TypeError:
                    pass

            self.baseline_values.append(np.std(y))

    def affine_transformation(self, t, t0=None, tn=None, x0=None, xn=None):

        t0 = t0 or min(self.baseline_values)
        tn = tn or max(self.baseline_values)
        x0 = x0 or 48
        xn = xn or 60

        x = (((xn-x0)/(tn-t0))*t+((x0*tn-xn*t0)/(tn-t0)))
        return x

    def start_midi_stream(self):

        midiout = rtmidi.MidiOut()
        available_ports = midiout.get_ports()

        if available_ports:
            midiout.open_port(0)
        else:
            midiout.open_virtual_port("My virtual output")

        print("Started MIDI stream...")

        while self.running:

            y = []

            for _ in range(self.fq):

                time.sleep(self.sleep_time)
                chunk = self.lsl.get_next_chunk()[0]
                try:
                    value = chunk[0, self.eeg_data.channels_list.index("Fp1")]
                    y.append(value)
                except TypeError:
                    pass

            note = int(self.affine_transformation(np.std(y)))
            print(note)

            midiout.send_message([0x90, note, 112])
            time.sleep(1)
            midiout.send_message([0x80, note, 0])


if __name__ == "__main__":
    lmi = LMInterface()
