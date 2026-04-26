#!/usr/bin/env python3
import socket
import threading
import pickle
import struct
import time
import psutil
import numpy as np
from flask import Flask, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# Configuration
EDGE_SOCKET_BIND = "0.0.0.0"
EDGE_SOCKET_PORT = 5001
METRICS_HTTP_PORT = 8000

# FL State
global_model_weights = None
local_model_weights = None
edge_aggregated_weights = None
device_weights = {}
device_samples = {}
fl_round = 0
training_status = 'idle'
training_lock = threading.Lock()

# Data storage
local_data = None
local_labels = None

# Network monitoring
_last_net = None
_last_time = time.time()

# Initialize CPU measurement
psutil.cpu_percent(interval=0.1)

# FL Model - Fixed with reshape instead of view
class FLModel(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # FIX: Use reshape instead of view
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

fl_model = FLModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fl_model.parameters(), lr=0.001)

app = Flask(__name__)

@app.route('/metrics', methods=['GET'])
def metrics_http():
    """HTTP endpoint for metrics - used by cloudcore"""
    # Get real CPU usage
    cpu = float(psutil.cpu_percent(interval=0.5))
    mem = float(psutil.virtual_memory().percent)
    
    now = time.time()
    io = psutil.net_io_counters()
    global _last_net, _last_time
    
    if _last_net is None:
        _last_net = io
        _last_time = now
        bw_mbps = 0.0
    else:
        elapsed = now - _last_time
        bytes_delta = (io.bytes_sent - _last_net.bytes_sent) + (io.bytes_recv - _last_net.bytes_recv)
        bw_mbps = (bytes_delta * 8) / (elapsed * 1024 * 1024) if elapsed > 0 else 0.0
        _last_net = io
        _last_time = now
    
    bw_norm = max(0.0, min(1.0, bw_mbps / 1000.0))
    
    return jsonify({
        "cpu": cpu,
        "mem": mem,
        "bw_used_mbps": bw_mbps,
        "bw_norm": bw_norm,
        'timestamp': now
    })

def start_metrics_http():
    app.run(host='0.0.0.0', port=METRICS_HTTP_PORT, debug=False, use_reloader=False)

def send_msg(sock, data):
    msg = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(msg)) + msg)

def recv_msg(sock):
    try:
        raw_len = sock.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack('>I', raw_len)[0]
        data = b''
        while len(data) < msg_len:
            chunk = sock.recv(min(msg_len - len(data), 65536))
            if not chunk:
                return None
            data += chunk
        return pickle.loads(data)
    except Exception as e:
        return None

def metrics_values():
    """Get metrics for socket communication"""
    cpu = float(psutil.cpu_percent(interval=0.5))
    mem = float(psutil.virtual_memory().percent)
    
    global _last_net, _last_time
    now = time.time()
    io = psutil.net_io_counters()
    
    if _last_net is None:
        _last_net = io
        _last_time = now
        bw_mbps = 0.0
    else:
        elapsed = now - _last_time
        if elapsed <= 0:
            bw_mbps = 0.0
        else:
            bytes_delta = (io.bytes_sent - _last_net.bytes_sent) + (io.bytes_recv - _last_net.bytes_recv)
            bw_mbps = (bytes_delta * 8) / (elapsed * 1024 * 1024)
        _last_net = io
        _last_time = now
    
    bw_norm = max(0.0, min(1.0, bw_mbps / 1000.0))
    
    return {
        "cpu": cpu,
        "mem": mem,
        "bw_mbps": bw_mbps,
        "bw_norm": bw_norm,
        "timestamp": now
    }

def aggregate_device_weights():
    """Federated Averaging at Edge Level"""
    if not device_weights:
        return None
    
    total_samples = sum(device_samples.values())
    aggregated_weights = []
    
    for i in range(len(next(iter(device_weights.values())))):
        layer_sum = None
        for device_id, weights in device_weights.items():
            num_samples = device_samples.get(device_id, 0)
            if layer_sum is None:
                layer_sum = weights[i] * num_samples
            else:
                layer_sum += weights[i] * num_samples
        aggregated_weights.append(layer_sum / total_samples)
    
    return aggregated_weights

def train_fl_model(round_num):
    global local_model_weights, training_status, edge_aggregated_weights
    
    with training_lock:
        training_status = 'training'
    
    try:
        print(f"🔄 Starting FL training on edge for round {round_num}")
        
        # Load appropriate weights - FIXED with proper cloning
        if device_weights:
            edge_aggregated_weights = aggregate_device_weights()
            if edge_aggregated_weights:
                # Create new tensors to avoid memory issues
                new_state_dict = {}
                for i, (name, param) in enumerate(fl_model.state_dict().items()):
                    if i < len(edge_aggregated_weights):
                        new_state_dict[name] = edge_aggregated_weights[i].clone()
                fl_model.load_state_dict(new_state_dict)
                print("✅ Loaded aggregated device weights")
        elif global_model_weights:
            # Create new tensors to avoid memory issues
            new_state_dict = {}
            for i, (name, param) in enumerate(fl_model.state_dict().items()):
                if i < len(global_model_weights):
                    new_state_dict[name] = global_model_weights[i].clone()
            fl_model.load_state_dict(new_state_dict)
            print("✅ Loaded global model weights")
        else:
            print("⚠️  No weights available, using random initialization")
        
        if local_data is not None and len(local_data) > 0:
            # Convert data to PyTorch tensors - FIXED with proper reshaping
            x_tensor = torch.FloatTensor(np.transpose(local_data, (0, 3, 1, 2)))  # NHWC to NCHW
            y_tensor = torch.LongTensor(local_labels)
            
            # Ensure proper tensor contiguity
            x_tensor = x_tensor.contiguous()
            
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training loop
            fl_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for epoch in range(2):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    output = fl_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                
                total_loss += epoch_loss
                print(f"   Epoch {epoch+1}/2 - Loss: {epoch_loss/len(dataloader):.4f}")
            
            accuracy = correct / total if total > 0 else 0
            avg_loss = total_loss / (len(dataloader) * 2)
            
            # Save trained weights - FIXED with proper cloning
            local_model_weights = [param.data.clone() for param in fl_model.parameters()]
            
            print(f"✅ Edge FL training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            with training_lock:
                training_status = 'done'
            
            return avg_loss, accuracy
        else:
            print("⚠️  No local data available for training")
            # If no data, just return current weights without training
            local_model_weights = [param.data.clone() for param in fl_model.parameters()]
            with training_lock:
                training_status = 'done'
            return 0.0, 0.0
            
    except Exception as e:
        print(f"❌ FL training error: {e}")
        import traceback
        traceback.print_exc()
        with training_lock:
            training_status = 'error'
        return 0.0, 0.0

def handle_request(client_sock, addr):
    global local_data, local_labels, global_model_weights, device_weights, device_samples, fl_round
    
    try:
        data = recv_msg(client_sock)
        if not data:
            client_sock.close()
            return
            
        cmd = data.get('cmd', '')
        
        if cmd == 'get_metrics':
            m = metrics_values()
            send_msg(client_sock, {'metrics': [m['cpu'], m['mem'], m['bw_norm']]})
            
        elif cmd == 'receive_data':
            local_data = np.array(data['data'], dtype=np.float32)
            local_labels = np.array(data['labels'], dtype=np.int64)
            device_id = data.get('device_id', -1)
            print(f"📥 Received {len(local_data)} samples from device {device_id}")
            
            # Process the data
            print("💾 Processing received data...")
            time.sleep(1)  # Simulate processing time
            
            send_msg(client_sock, {'status': 'data_received'})
            
        elif cmd == 'update_global_model':
            global_model_weights = data['weights']
            fl_round = data.get('fl_round', 0)
            print(f"🔄 Updated global model for FL round {fl_round}")
            send_msg(client_sock, {'status': 'global_model_updated'})
            
        elif cmd == 'start_fl':
            round_num = data.get('round', 0)
            fl_round = data.get('fl_round', round_num)
            
            if training_status != 'training':
                threading.Thread(target=train_fl_model, args=(round_num,), daemon=True).start()
                send_msg(client_sock, {'status': 'fl_training_started', 'round': round_num})
            else:
                send_msg(client_sock, {'status': 'already_training'})
                
        elif cmd == 'check_status':
            with training_lock:
                status_response = {'status': training_status}
                if training_status == 'done' and local_data is not None:
                    status_response['data_samples'] = len(local_data)
                send_msg(client_sock, status_response)
                
        elif cmd == 'get_fl_weights':
            # Calculate current metrics
            loss, accuracy = 0.0, 0.0
            if local_data is not None and len(local_data) > 0:
                x_tensor = torch.FloatTensor(np.transpose(local_data, (0, 3, 1, 2)))
                y_tensor = torch.LongTensor(local_labels)
                
                # Ensure tensor contiguity
                x_tensor = x_tensor.contiguous()
                
                fl_model.eval()
                with torch.no_grad():
                    output = fl_model(x_tensor)
                    loss = criterion(output, y_tensor).item()
                    _, predicted = torch.max(output.data, 1)
                    accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            
            # Send appropriate weights - FIXED with proper cloning
            if local_model_weights is not None:
                weights_to_send = [w.clone() for w in local_model_weights]
            elif global_model_weights is not None:
                weights_to_send = [w.clone() for w in global_model_weights]
            else:
                weights_to_send = [param.data.clone() for param in fl_model.parameters()]
            
            response = {
                'weights': weights_to_send,
                'num_samples': len(local_data) if local_data is not None else 0,
                'metrics': {
                    'loss': loss,
                    'accuracy': accuracy
                },
                'fl_round': fl_round
            }
            
            send_msg(client_sock, response)
            print(f"📤 Sent FL weights to cloud - Round {fl_round}, Accuracy: {accuracy:.4f}")
            
        elif cmd == 'submit_device_weights':
            device_id = data['device_id']
            device_weights[device_id] = data['weights']
            device_samples[device_id] = data['num_samples']
            print(f"📊 Received weights from device {device_id} ({data['num_samples']} samples)")
            send_msg(client_sock, {'status': 'weights_received'})
            
        else:
            send_msg(client_sock, {'error': f'unknown_cmd: {cmd}'})
            
    except Exception as e:
        print(f"Request handling error from {addr}: {e}")
    finally:
        try:
            client_sock.close()
        except:
            pass

def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((EDGE_SOCKET_BIND, EDGE_SOCKET_PORT))
    server.listen(10)
    
    print(f"🖥️  Edge FL Server listening on {EDGE_SOCKET_BIND}:{EDGE_SOCKET_PORT}")
    print("✅ Ready for Federated Learning with devices")
    print("📊 Metrics available at http://<edge_ip>:8000/metrics")
    
    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_request, args=(client, addr), daemon=True).start()

if __name__ == '__main__':
    print("🚀 Starting Enhanced Edge Node")
    print("💻 Real-time CPU metrics enabled")
    print("🤖 FL Model: CNN for Fashion-MNIST")
    threading.Thread(target=start_metrics_http, daemon=True).start()
    start_socket_server()


