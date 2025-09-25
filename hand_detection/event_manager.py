class EventManager:
    def __init__(self):
        self._listeners = {}

    def add_listener(self, event_name, callback):
        """Register a listener for a specific event."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    def trigger_event(self, event_name, *args, **kwargs):
        """Trigger an event and notify all listeners."""
        if event_name in self._listeners:
            for callback in self._listeners[event_name]:
                callback(*args, **kwargs)


# global instance of the EventManager
event_manager = EventManager()
