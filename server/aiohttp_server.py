import socketio
from aiohttp import web

if __name__ == '__main__':
    sio = socketio.AsyncServer(async_mode='aiohttp')
    app = web.Application()
    sio.attach(app)
    web.run_app(app)