import threading
import sys

# A coordinator object that can be used to alternate execution
# between the main thread and another thread
class Coordinator:
    def __init__(self, main_turn_first = True):
        self.condition = threading.Condition()
        self.main_turn = main_turn_first
        self.running = True

    def wait_for_turn(self, is_main):
        with self.condition:
            while self.running and (self.main_turn != is_main):
                self.condition.wait()
        if not self.running and not is_main:
            sys.exit(0)

    def switch_turn(self):
        with self.condition:
            self.main_turn = not self.main_turn
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
    
    def restart(self, main_turn_first = True):
        with self.condition:
            self.main_turn = main_turn_first
            self.running = True
