# Contains info for a pod
class Pod:
    def __init__(self, name, cpu, memory, duration, arrival_time):
        self.name = name
        self.cpu = cpu
        self.memory = memory
        self.duration = duration
        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None
        self.node = None

    def __eq__(self, other):
        return isinstance(other, Pod) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
    
    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return f"Pod(name={self.name}, cpu={self.cpu}, mem={self.memory}, duration={self.duration}, arrival_time={self.arrival_time})"
