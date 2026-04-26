#!/usr/bin/env python3
# device.py - Fixed for PyTorch compatibility
import socket
import pickle
import numpy as np
import time
import struct
import threading
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

# Configuration
CLOUD_IP = '192.168.92.20'
CLOUD_PORT = 5000
DEVICE_IP = '0.0.0.0'
DEVICE_PORT = 5002
ROUNDS = 50
DEVICE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print(f"📦 Loading Fashion-MNIST for Device {DEVICE_ID}...")

# Load Fashion-MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    all_data, all_labels = next(iter(trainloader))
    all_data = all_data.numpy()
    all_labels = all_labels.numpy()
    
    # Split data among devices (each gets 1000 samples)
    start_idx = DEVICE_ID * 1000
    end_idx = start_idx + 1000
    device_data = all_data[start_idx:end_idx]
    device_labels = all_labels[start_idx:end_idx]
    
    # Convert to NHWC format for compatibility
    device_data = np.transpose(device_data, (0, 2, 3, 1))  # NCHW to NHWC
    
    print(f"✅ Loaded {len(device_data)} Fashion-MNIST samples for Device {DEVICE_ID}")
    
except Exception as e:
    print(f"❌ Error loading Fashion-MNIST: {e}")
    # Create synthetic data if loading fails
    device_data = np.random.rand(1000, 28, 28, 1).astype(np.float32)
    device_labels = np.random.randint(0, 10, 1000)
    print("⚠️  Using synthetic data for testing")

# Device FL Model - MUST match edge architecture exactly
class DeviceFLModel(nn.Module):
    def __init__(self):
        super(DeviceFLModel, self).__init__()
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

device_model = DeviceFLModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(device_model.parameters(), lr=0.001)

# Assignment tracking
assignment_received = threading.Event()
assignment_data = {}
current_fl_round = 0

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

def send_to(ip, port, data, timeout=20):
    try:
        with socket.socket() as sock:
            sock.settimeout(timeout)
            sock.connect((ip, port))
            send_msg(sock, data)
            return recv_msg(sock)
    except Exception as e:
        print(f"Send error to {ip}:{port}: {e}")
        return None

def send_ready_signal(round_num):
    print(f"📡 Round {round_num}: Sending ready signal to Cloud...")
    response = send_to(CLOUD_IP, CLOUD_PORT, {
        'cmd': 'ready', 
        'round': round_num, 
        'device_id': DEVICE_ID,
        'timestamp': time.time()
    })
    return response and response.get('status') == 'acknowledged'

def send_data_offloaded_signal(round_num):
    """Notify cloud that data offloading is complete"""
    response = send_to(CLOUD_IP, CLOUD_PORT, {
        'cmd': 'data_offloaded', 
        'device_id': DEVICE_ID,
        'round': round_num,
        'timestamp': time.time()
    })
    return response and response.get('status') == 'acknowledged'

def train_local_model():
    """Train local model on device data"""
    if len(device_data) == 0:
        return None, 0
    
    # Convert to PyTorch tensors (NHWC to NCHW)
    x_tensor = torch.FloatTensor(np.transpose(device_data, (0, 3, 1, 2)))
    y_tensor = torch.LongTensor(device_labels)
    
    # Ensure tensor contiguity
    x_tensor = x_tensor.contiguous()
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Local training
    device_model.train()
    total_loss = 0
    
    for epoch in range(2):  # 2 epochs for faster training
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = device_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        total_loss += epoch_loss
        print(f"   Device {DEVICE_ID} - Epoch {epoch+1}/2 - Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Get updated weights
    local_weights = [param.data.clone() for param in device_model.parameters()]
    
    avg_loss = total_loss / (len(dataloader) * 2)
    print(f"🔧 Device {DEVICE_ID} local training completed - Avg Loss: {avg_loss:.4f}")
    return local_weights, len(device_data)

def offload_data_and_weights(edge_ip, edge_idx, round_num, fl_round):
    print(f"📤 Offloading data and local model to Edge {edge_idx} ({edge_ip}) for Round {round_num}...")
    
    # First, send the data to edge
    data_payload = {
        'cmd': 'receive_data',
        'data': device_data,
        'labels': device_labels,
        'round': round_num,
        'device_id': DEVICE_ID,
        'fl_round': fl_round,
        'timestamp': time.time()
    }
    
    data_response = send_to(edge_ip, 5001, data_payload, timeout=60)
    if not data_response or data_response.get('status') != 'data_received':
        print(f"❌ Data offload failed for round {round_num}")
        return False
    
    print(f"✅ Data offload successful for round {round_num}")
    
    # Train local model and send weights to edge
    print(f"🔧 Training local model on device {DEVICE_ID} for round {round_num}...")
    local_weights, num_samples = train_local_model()
    
    if local_weights is not None:
        weights_payload = {
            'cmd': 'submit_device_weights',
            'weights': local_weights,
            'num_samples': num_samples,
            'device_id': DEVICE_ID,
            'round': round_num,
            'fl_round': fl_round,
            'timestamp': time.time()
        }
        
        weights_response = send_to(edge_ip, 5001, weights_payload, timeout=30)
        if weights_response and weights_response.get('status') == 'weights_received':
            print(f"✅ Local weights submitted to edge for FL round {fl_round}")
            
            # Notify cloud that data offloading is complete
            if send_data_offloaded_signal(round_num):
                print(f"✅ Cloud notified about data offloading completion")
                return True
            else:
                print(f"⚠️  Failed to notify cloud about data offloading")
                return True
        else:
            print(f"⚠️  Weight submission failed for FL round {fl_round}")
            return False
    else:
        print(f"⚠️  Local training failed, no weights to submit")
        return False

def handle_assignment(client_sock):
    global assignment_data, current_fl_round
    try:
        data = recv_msg(client_sock)
        if data and data['cmd'] == 'assign':
            assigned_round = data['round']
            current_fl_round = data.get('fl_round', assigned_round)
            
            print(f"✅ Round {assigned_round}: Assigned to Edge {data['edge_idx']} ({data['edge_ip']}) for FL Round {current_fl_round}")
            send_msg(client_sock, {'status': 'ok', 'device_id': DEVICE_ID, 'round': assigned_round})
            assignment_data = data
            assignment_received.set()
    except Exception as e:
        print(f"Assignment handling error: {e}")
    finally:
        client_sock.close()

def start_listener():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((DEVICE_IP, DEVICE_PORT))
    server.listen(5)
    print(f"📱 Device {DEVICE_ID} listening for assignments on port {DEVICE_PORT}")
    print(f"🤖 Ready for Federated Learning participation")
    print(f"📊 Local data: {len(device_data)} Fashion-MNIST samples")
    
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle_assignment, args=(client,), daemon=True).start()

if __name__ == '__main__':
    threading.Thread(target=start_listener, daemon=True).start()
    time.sleep(3)
    
    current_round = 1
    
    while current_round <= ROUNDS:
        print(f"\n{'='*50}")
        print(f"Device {DEVICE_ID} - Round {current_round}/{ROUNDS}")
        print(f"{'='*50}")
        
        assignment_received.clear()
        assignment_data.clear()
        
        # Try sending ready signal with retries
        ready_success = False
        for attempt in range(3):
            if send_ready_signal(current_round):
                ready_success = True
                break
            else:
                print(f"⚠️  Ready signal attempt {attempt+1}/3 failed, retrying in 5s...")
                time.sleep(5)
        
        if not ready_success:
            print(f"❌ Could not send ready signal for round {current_round}")
            current_round += 1
            time.sleep(10)
            continue
        
        # Wait for assignment with timeout
        if not assignment_received.wait(timeout=60):
            print(f"⚠️  Round {current_round}: No assignment from Cloud within 60s")
            current_round += 1
            time.sleep(5)
            continue
        
        # Offload data and weights
        success = offload_data_and_weights(
            assignment_data['edge_ip'], 
            assignment_data['edge_idx'], 
            current_round, 
            current_fl_round
        )
        
        if success:
            print(f"✅ Round {current_round} FL participation complete")
        else:
            print(f"⚠️  Round {current_round} FL participation had issues")
        
        # Move to next round
        current_round += 1
        
        print(f"⏳ Waiting for next round to start...")
        time.sleep(20)  # Wait between rounds
    
    print(f"\n🎉 Device {DEVICE_ID}: Completed all {ROUNDS} FL rounds!")


