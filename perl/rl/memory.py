

class RingBuffer:
    """
    Ringbuffer is a buffer with fixed capacity in the form of a ring:
    when the buffer is full, the oldest value is removed
    """
    def __init__(self, capacity):
        """
        Create Ringbuffer with <capacity>
        """
        self.capacity = capacity
        self.length = 0

        self._content = []
        self._index = 0

    def add(self, value):
        """
        Add new value to the buffer
        """
        if self.length < self.capacity:
            self._content.append(value)
            self.length += 1
        else:
            self._content[self._index] = value
            self._index = (self._index + 1) % self.capacity

    def __iter__(self):
        return self._content.__iter__()

    def __len__(self):
        return self.length





