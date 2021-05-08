import socketio

HOST = '127.0.0.1'
PORT = 54325

static_files = {
    '/files': './imgs',
    '/': 'README.md'
}

# create a Socket.IO server
sio = socketio.AsyncServer()

# wrap with ASGI application
app = socketio.ASGIApp(sio, static_files=static_files)

@sio.event
async def my_event(sid, data):
    pass

@sio.on('my custom event')
async def another_event(sid, data):
    pass

@sio.event
async def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)