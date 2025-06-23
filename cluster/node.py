# Contains info for a cluster node
class Node:
    def __init__(self, name, cpu_capacity, mem_capacity):
        self.name = name
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.cpu_available = cpu_capacity
        self.mem_available = mem_capacity

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Node(name={self.name}, cpu={self.cpu_capacity}, mem={self.mem_capacity})"

    def allocate_resources(self, cpu, memory):
        self.cpu_available -= cpu
        self.mem_available -= memory
        if self.cpu_available < 0 : self.cpu_available = 0
        if self.mem_available < 0 : self.mem_available = 0

    def release_resources(self, cpu, memory):
        self.cpu_available += cpu
        self.mem_available += memory
        if self.cpu_available > self.cpu_capacity : self.cpu_available = self.cpu_capacity
        if self.mem_available > self.mem_capacity : self.mem_available = self.mem_capacity

    def has_available_resources(self, cpu, memory) -> bool:
        return self.cpu_available >= cpu and self.mem_available >= memory
