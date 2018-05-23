import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class ProgressBar:

    def __init__(self, total, counting=True):
        self.total = total
        self.progress = 0
        self.counting = counting

    def update_progressbar(self, message):
        """
        Displays or updates a console progress bar.
        """

        if self.counting:
            self.progress += 1

            barLength, status = 20, message  # ""
            progress = float(self.progress) / float(self.total)
            if progress >= 1.:
                progress, status = 1, "\r\n" + message
            block = int(round(barLength * progress))
            text = "\r[{}] {:.0f}% {}".format(
                "#" * block + "-" * (barLength - block), round(progress * 100, 0),
                status)
            sys.stdout.write(text)
            sys.stdout.flush()


if __name__ == "__main__":

    runs = 300
    message = "a{}"

    pb = ProgressBar(runs)

    for run_num in range(runs):
        time.sleep(.01)
        # message += "1"
        pb.update_progressbar(message)
