import cherrypy

class StateServer:
    def __init__(self):
        self.state = "idle"  # Initial state value

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def status(self):
        """Returns the current state."""
        return {"state": self.state}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def update(self):
        """Updates the state value based on input JSON."""
        input_data = cherrypy.request.json
        new_state = input_data.get("state", None)
        
        if new_state:
            self.state = new_state
            return {"status": "success", "new_state": self.state}
        else:
            return {"status": "error", "message": "Invalid state value"}

if __name__ == '__main__':
    cherrypy.config.update({
        'server.socket_host': '0.0.0.0',
        'server.socket_port': 8080
    })
    cherrypy.quickstart(StateServer())
