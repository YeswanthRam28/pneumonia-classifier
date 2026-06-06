import http.server
import socketserver
import urllib.request
import urllib.error
import json

PORT = 3000

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Forward the request to Nvidia's API
            req = urllib.request.Request(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                data=post_data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': self.headers.get('Authorization', '')
                }
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    body = response.read()
                    self.send_response(response.status)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(body)
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(e.read())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        print(f"PNEUMA Local Server running at http://localhost:{PORT}")
        print("This server bypasses browser CORS policies for the Nvidia API.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
