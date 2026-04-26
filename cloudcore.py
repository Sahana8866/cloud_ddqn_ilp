#!/usr/bin/env python3
# cloudcore.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import socket
import pickle
import threading
import time
from collections import deque, namedtuple
import random
import struct
import requests
import os

# ==================== CONFIG ====================
CLOUD_IP = '192.168.92.20'
CLOUD_PORT = 5000
EDGE_IPS = ['192.168.92.21', '192.168.92.22', '192.168.92.23']
DEVICE_IPS = ['192.168.92.24', '192.168.92.26']
NUM_EDGES = len(EDGE_IPS)
NUM_DEVICES = len(DEVICE_IPS)
STATE_SIZE = NUM_EDGES * 4  # CHANGED: Now 4 metrics per edge (cpu, mem, bw, stress)
ACTION_SIZE = NUM_EDGES
ROUNDS = 50
BATCH_SIZE = 16
BUFFER_SIZE = 2000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99
LR = 0.0005
TARGET_UPDATE = 10
WARMUP_ROUNDS = 5
LAMBDA = [0.5, 0.4, 0.1]  # latency, energy, cpu

# Stress scenarios
STRESS_SCENARIOS = {
    'NO_STRESS': {'cpu_range': (5, 25), 'mem_range': (15, 30), 'bw_range': (0.01, 0.1)},
    'LOW_STRESS': {'cpu_range': (20, 40), 'mem_range': (25, 45), 'bw_range': (0.1, 0.3)},
    'MED_STRESS': {'cpu_range': (40, 70), 'mem_range': (40, 65), 'bw_range': (0.3, 0.6)},
    'HIGH_STRESS': {'cpu_range': (70, 95), 'mem_range': (65, 90), 'bw_range': (0.6, 0.9)}
}

# Global state
edge_stress_levels = {i: 'NO_STRESS' for i in range(NUM_EDGES)}
current_round = 0
round_in_progress = threading.Event()
device_ready = {}
edge_training_done = {}
edge_metrics_cache = {}
global_model_weights = None
fl_round = 0
fl_accuracy_history = []
fl_loss_history = []

class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

policy_net = DDQN()
target_net = DDQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
memory = deque(maxlen=BUFFER_SIZE)

def create_fl_model():
    """Create a simple CNN model for Fashion-MNIST"""
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def aggregate_fl_weights(edge_weights_list):
    """Federated Averaging at Cloud Level"""
    if not edge_weights_list:
        return global_model_weights
    
    total_samples = sum(num_samples for _, num_samples in edge_weights_list)
    
    averaged_weights = []
    for i in range(len(edge_weights_list[0][0])):
        layer_sum = None
        for weights, num_samples in edge_weights_list:
            if layer_sum is None:
                layer_sum = weights[i] * num_samples
            else:
                layer_sum += weights[i] * num_samples
        averaged_weights.append(layer_sum / total_samples)
    
    return averaged_weights

def save_global_model(round_num, accuracy=0.0, loss=0.0):
    """Save the global model to disk"""
    if global_model_weights is None:
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    model_data = {
        'weights': global_model_weights,
        'round': round_num,
        'accuracy': accuracy,
        'loss': loss,
        'timestamp': time.time(),
        'model_architecture': 'CNN-FashionMNIST'
    }
    
    filename = f"saved_models/global_model_round_{round_num}.pth"
    torch.save(model_data, filename)
    print(f"💾 Global model saved to {filename}")

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

def send_to_socket(ip, port, data, timeout=5):
    try:
        with socket.socket() as sock:
            sock.settimeout(timeout)
            sock.connect((ip, port))
            send_msg(sock, data)
            resp = recv_msg(sock)
            return resp
    except Exception:
        return None

def get_real_edge_metrics(edge_ip, edge_idx):
    """Get real metrics from edges with stress simulation"""
    # Try to get actual metrics first
    resp = send_to_socket(edge_ip, 5001, {'cmd': 'get_metrics'}, timeout=2)
    if resp and isinstance(resp, dict) and 'metrics' in resp:
        metrics = resp['metrics']
        cpu = float(metrics[0])
        mem = float(metrics[1])
        bw_norm = float(metrics[2])
        cpu = max(0.0, min(100.0, cpu))
        mem = max(0.0, min(100.0, mem))
        bw_norm = max(0.0, min(1.0, bw_norm))
        return [cpu, mem, bw_norm]
    
    # Fallback to stress-based simulation
    stress_level = edge_stress_levels[edge_idx]
    scenario = STRESS_SCENARIOS[stress_level]
    
    cpu = random.uniform(*scenario['cpu_range'])
    mem = random.uniform(*scenario['mem_range'])
    bw_norm = random.uniform(*scenario['bw_range'])
    
    print(f"📊 Edge {edge_idx} ({edge_ip}): {stress_level} - CPU={cpu:.1f}%, Mem={mem:.1f}%, BW={bw_norm:.3f}")
    return [cpu, mem, bw_norm]

def get_state():
    """FIXED: State now includes stress information"""
    state = []
    for i, edge_ip in enumerate(EDGE_IPS):
        metrics = get_real_edge_metrics(edge_ip, i)
        edge_metrics_cache[edge_ip] = metrics
        cpu_norm = metrics[0] / 100.0
        mem_norm = metrics[1] / 100.0
        bw_norm = metrics[2]
        
        # CRITICAL FIX: Add stress level to state
        stress_level = edge_stress_levels[i]
        stress_map = {'NO_STRESS': 0.0, 'LOW_STRESS': 0.33, 'MED_STRESS': 0.66, 'HIGH_STRESS': 1.0}
        stress_norm = stress_map[stress_level]
        
        state.extend([cpu_norm, mem_norm, bw_norm, stress_norm])  # Now 4 metrics per edge
    
    return np.array(state, dtype=np.float32)

def update_stress_scenarios(round_num):
    """Update stress levels randomly for edges"""
    if round_num <= WARMUP_ROUNDS:
        return  # No stress during warmup
    
    stress_levels = list(STRESS_SCENARIOS.keys())
    
    for edge_idx in range(NUM_EDGES):
        # 30% chance to change stress level each round after warmup
        if random.random() < 0.3:
            new_stress = random.choice(stress_levels)
            if new_stress != edge_stress_levels[edge_idx]:
                edge_stress_levels[edge_idx] = new_stress
                print(f"🔥 Edge {edge_idx} stress changed to: {new_stress}")

def calculate_realistic_metrics(cpu_norm, mem_norm, bw_norm, stress_level):
    """FIXED: Calculate realistic latency and energy with stress impact"""
    # Base metrics
    base_latency = 50.0
    base_energy = 20.0
    
    # Load factor
    load_factor = (cpu_norm * 0.6 + mem_norm * 0.3 + (1 - bw_norm) * 0.1)
    
    # Stress multipliers - FIXED: Higher impact
    stress_multipliers = {
        'NO_STRESS': 1.0,
        'LOW_STRESS': 1.5, 
        'MED_STRESS': 2.0,
        'HIGH_STRESS': 3.0
    }
    stress_mult = stress_multipliers[stress_level]
    
    # Calculate with stress impact
    latency = base_latency + (load_factor * 300.0 * stress_mult)
    latency = min(latency, 1000.0)  # Increased max
    
    energy = base_energy + (load_factor * 60.0 * stress_mult)
    energy = min(energy, 150.0)  # Increased max
    
    return latency, energy

def ilp_assign(candidates_per_device, state):
    """ILP Assignment with realistic costs"""
    assignments = [-1] * NUM_DEVICES
    available_edges = set(range(NUM_EDGES))
    
    print(f"  ILP Assignment with current stress levels:")
    for edge_idx in range(NUM_EDGES):
        print(f"    Edge {edge_idx}: {edge_stress_levels[edge_idx]}")

    for dev_id in range(NUM_DEVICES):
        candidates = [e for e in candidates_per_device[dev_id] if e in available_edges]
        if not candidates:
            candidates = list(available_edges)
        if not candidates:
            candidates = list(range(NUM_EDGES))

        costs = {}
        for e in candidates:
            # FIXED: Get metrics from state (4 metrics per edge now)
            cpu_norm = state[e*4]      # CPU at position 0,4,8...
            mem_norm = state[e*4+1]    # Memory at 1,5,9...
            bw_norm = state[e*4+2]     # Bandwidth at 2,6,10...
            stress_norm = state[e*4+3] # Stress at 3,7,11...
            
            # Convert stress norm back to level
            stress_level = 'NO_STRESS'
            if stress_norm > 0.75:
                stress_level = 'HIGH_STRESS'
            elif stress_norm > 0.5:
                stress_level = 'MED_STRESS'
            elif stress_norm > 0.25:
                stress_level = 'LOW_STRESS'
            
            latency, energy = calculate_realistic_metrics(cpu_norm, mem_norm, bw_norm, stress_level)
            cpu_actual = cpu_norm * 100.0
            
            # Normalize for cost calculation
            latency_norm = latency / 1000.0  # Max 1000ms
            energy_norm = energy / 150.0     # Max 150J
            cpu_norm_cost = cpu_norm
            
            # Base cost
            base_cost = LAMBDA[0] * latency_norm + LAMBDA[1] * energy_norm + LAMBDA[2] * cpu_norm_cost
            
            # Stress penalty
            stress_penalty = 0.0
            if edge_stress_levels[e] == 'LOW_STRESS':
                stress_penalty = 0.2
            elif edge_stress_levels[e] == 'MED_STRESS':
                stress_penalty = 0.5
            elif edge_stress_levels[e] == 'HIGH_STRESS':
                stress_penalty = 1.0
                
            costs[e] = base_cost + stress_penalty
            
            print(f"    Edge {e}: CPU={cpu_actual:.1f}% -> Latency={latency:.1f}ms, Energy={energy:.1f}J, Stress={edge_stress_levels[e]} -> Cost={costs[e]:.3f}")

        best_edge = min(candidates, key=lambda e: costs.get(e, 999.0))
        assignments[dev_id] = best_edge
        available_edges.discard(best_edge)
        print(f"  Device {dev_id} → Edge {best_edge} (cost={costs.get(best_edge, 999.0):.3f})")

    return assignments

def select_action(state_tensor, round_num):
    """FIXED: Proper action selection without collisions"""
    global EPSILON
    
    if random.random() < EPSILON or round_num <= WARMUP_ROUNDS:
        print(f"  DDQN: Exploration (ε={EPSILON:.3f})")
        # Return empty candidates to force ILP
        return [[] for _ in range(NUM_DEVICES)], True
    else:
        with torch.no_grad():
            q_vals = policy_net(state_tensor).numpy().flatten()
        
        print(f"  DDQN: Q-values: {[f'{q:.3f}' for q in q_vals]}")
        
        # FIXED: Simple assignment without collisions
        assignments = []
        available_edges = list(range(NUM_EDGES))
        random.shuffle(available_edges)  # Randomize to avoid bias
        
        # Assign each device to a different edge
        for dev_idx in range(NUM_DEVICES):
            if available_edges:
                assigned_edge = available_edges.pop(0)
                assignments.append(assigned_edge)
            else:
                # If no edges left, use the one with highest Q-value
                best_edge = np.argmax(q_vals)
                assignments.append(best_edge)
        
        candidates = [[a] for a in assignments]
        
        print(f"  DDQN: Direct assignment {assignments}")
        return candidates, False

def train_ddqn():
    """FIXED: Proper DDQN training with gradient clipping"""
    if len(memory) < BATCH_SIZE:
        return
    
    batch = random.sample(memory, BATCH_SIZE)
    batch_t = Transition(*zip(*batch))
    
    # Filter out invalid actions
    valid_indices = [i for i, a in enumerate(batch_t.action) if 0 <= a < ACTION_SIZE]
    if not valid_indices:
        return
        
    state_batch = torch.FloatTensor(np.array([batch_t.state[i] for i in valid_indices]))
    action_batch = torch.LongTensor([batch_t.action[i] for i in valid_indices])
    reward_batch = torch.FloatTensor([batch_t.reward[i] for i in valid_indices])
    next_state_batch = torch.FloatTensor(np.array([batch_t.next_state[i] for i in valid_indices]))
    done_batch = torch.FloatTensor([batch_t.done[i] for i in valid_indices])

    # Current Q values
    q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    
    # Double DQN target
    with torch.no_grad():
        next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_state_batch).gather(1, next_actions).squeeze()
        target_q = reward_batch + GAMMA * next_q_values * (1.0 - done_batch)
    
    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    if random.random() < 0.1:  # Log only 10% of the time to reduce noise
        print(f"  DDQN Training: Loss={loss.item():.4f}, Memory={len(memory)}")

def distribute_global_model():
    """Send global model to all edges"""
    if global_model_weights is None:
        return
        
    print("🔄 Distributing global model to all edges...")
    success_count = 0
    for edge_ip in EDGE_IPS:
        resp = send_to_socket(edge_ip, 5001, {
            'cmd': 'update_global_model',
            'weights': global_model_weights,
            'fl_round': fl_round
        }, timeout=5)
        if resp and resp.get('status') == 'global_model_updated':
            success_count += 1
    
    print(f"✅ Global model distributed to {success_count}/{NUM_EDGES} edges")

def collect_fl_results(edge_assignments):
    """Collect FL results from edges and aggregate"""
    edge_weights_list = []
    total_accuracy = 0
    total_loss = 0
    valid_edges = 0
    
    for edge_idx in edge_assignments:
        edge_ip = EDGE_IPS[edge_idx]
        resp = send_to_socket(edge_ip, 5001, {'cmd': 'get_fl_weights'}, timeout=10)
        if resp and 'weights' in resp:
            weights = resp['weights']
            num_samples = resp.get('num_samples', 1000)
            metrics = resp.get('metrics', {})
            
            edge_weights_list.append((weights, num_samples))
            
            accuracy = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 1.0)
            total_accuracy += accuracy
            total_loss += loss
            valid_edges += 1
            
            print(f"  Edge {edge_idx}: FL Accuracy={accuracy:.4f}, Loss={loss:.4f}")
    
    if edge_weights_list:
        # Federated Averaging
        new_global_weights = aggregate_fl_weights(edge_weights_list)
        
        if valid_edges > 0:
            avg_accuracy = total_accuracy / valid_edges
            avg_loss = total_loss / valid_edges
            fl_accuracy_history.append(avg_accuracy)
            fl_loss_history.append(avg_loss)
            print(f"📊 FL Round {fl_round}: Avg Accuracy={avg_accuracy:.4f}, Avg Loss={avg_loss:.4f}")
        
        return new_global_weights, avg_accuracy, avg_loss
    return None, 0.0, 0.0

def calculate_reward(assignments, state):
    """FIXED: Proper reward calculation matching mathematical model"""
    if not assignments or all(a == -1 for a in assignments):
        return -10.0  # Heavy penalty for no assignment
    
    total_cost = 0.0
    valid_assignments = 0
    
    print("\n📊 REWARD CALCULATION:")
    for dev_idx, edge_idx in enumerate(assignments):
        if edge_idx != -1 and 0 <= edge_idx < NUM_EDGES:
            # FIXED: Get metrics from state (4 metrics per edge now)
            cpu_norm = state[edge_idx*4]      # CPU at position 0,4,8...
            mem_norm = state[edge_idx*4+1]    # Memory at 1,5,9...
            bw_norm = state[edge_idx*4+2]     # Bandwidth at 2,6,10...
            stress_norm = state[edge_idx*4+3] # Stress at 3,7,11...
            
            # Convert stress norm back to level
            stress_level = 'NO_STRESS'
            if stress_norm > 0.75:
                stress_level = 'HIGH_STRESS'
            elif stress_norm > 0.5:
                stress_level = 'MED_STRESS'
            elif stress_norm > 0.25:
                stress_level = 'LOW_STRESS'
            
            # Calculate metrics with stress impact
            latency, energy = calculate_realistic_metrics(cpu_norm, mem_norm, bw_norm, stress_level)
            cpu_actual = cpu_norm * 100.0
            
            # NORMALIZE properly (0-1 range)
            latency_norm = latency / 1000.0  # Max 1000ms
            energy_norm = energy / 150.0     # Max 150J
            cpu_norm_cost = cpu_norm         # Already 0-1
            
            # Calculate cost EXACTLY as per mathematical model
            base_cost = (LAMBDA[0] * latency_norm + 
                        LAMBDA[1] * energy_norm + 
                        LAMBDA[2] * cpu_norm_cost)
            
            # Additional stress penalty (separate from the metrics)
            stress_penalties = {
                'NO_STRESS': 0.0,
                'LOW_STRESS': 0.2,
                'MED_STRESS': 0.5, 
                'HIGH_STRESS': 1.0
            }
            stress_penalty = stress_penalties[stress_level]
            
            total_assign_cost = base_cost + stress_penalty
            total_cost += total_assign_cost
            valid_assignments += 1
            
            print(f"   Device {dev_idx}→Edge {edge_idx} ({stress_level}): "
                  f"CPU={cpu_actual:.1f}%, Latency={latency:.1f}ms, "
                  f"Energy={energy:.1f}J -> Cost={total_assign_cost:.3f}")
    
    if valid_assignments == 0:
        return -10.0
    
    avg_cost = total_cost / valid_assignments
    reward = -avg_cost  # Negative reward = minimizing cost
    
    print(f"   AVERAGE COST: {avg_cost:.3f}")
    print(f"   🎁 FINAL REWARD: {reward:.3f}")
    
    return reward

def run_round(round_num):
    global current_round, EPSILON, fl_round, global_model_weights
    current_round = round_num
    fl_round = round_num
    round_in_progress.set()
    
    print(f"\n{'='*60}")
    print(f"🎯 ROUND {round_num}/{ROUNDS} - FL Round {fl_round}")
    print(f"{'='*60}")
    
    # Update stress scenarios
    update_stress_scenarios(round_num)
    
    # Initialize global model if first round
    if global_model_weights is None and round_num == 1:
        model = create_fl_model()
        global_model_weights = [param.data.clone() for param in model.parameters()]
        print("✅ Initialized global FL model")
        save_global_model(0, 0.0, 0.0)  # Save initial model
    
    # Distribute global model
    distribute_global_model()
    
    # Get system state
    state = get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # Wait for devices
    print("⏳ Waiting for devices to be ready...")
    device_ready.clear()
    timeout = time.time() + 30
    while len(device_ready) < NUM_DEVICES and time.time() < timeout:
        time.sleep(1)
    
    if len(device_ready) < NUM_DEVICES:
        print(f"⚠️  Only {len(device_ready)}/{NUM_DEVICES} devices ready")
        round_in_progress.clear()
        return None, None, -5.0, state, False
    
    print(f"✅ All {NUM_DEVICES} devices ready")
    
    # Decision making
    if round_num <= WARMUP_ROUNDS:
        print("🔵 WARMUP: Using ILP only")
        candidates = [[e for e in range(NUM_EDGES)] for _ in range(NUM_DEVICES)]
        assignments = ilp_assign(candidates, state)
    else:
        print("🟢 DDQN+ILP Hybrid Decision")
        candidates, use_ilp = select_action(state_tensor, round_num)
        if use_ilp:
            assignments = ilp_assign(candidates, state)
        else:
            assignments = [c[0] for c in candidates]
    
    print(f"✅ Final Assignments: {assignments}")
    
    # Notify devices
    dev_ips = list(device_ready.keys())[:NUM_DEVICES]
    for dev_idx, dev_ip in enumerate(dev_ips):
        edge_idx = assignments[dev_idx]
        if edge_idx != -1:
            resp = send_to_socket(dev_ip, 5002, {
                'cmd': 'assign', 
                'edge_ip': EDGE_IPS[edge_idx], 
                'edge_idx': edge_idx, 
                'round': round_num,
                'fl_round': fl_round
            }, timeout=5)
    
    # Wait for data offload and local training
    print("⏳ Waiting for data offload and local training...")
    time.sleep(8)
    
    # Start FL training on assigned edges
    unique_edges = list(set([a for a in assignments if a != -1]))
    if not unique_edges:
        print("❌ No valid edges for FL training")
        round_in_progress.clear()
        return None, None, -5.0, state, False
    
    print(f"🤖 Starting FL training on edges: {unique_edges}")
    for edge_idx in unique_edges:
        send_to_socket(EDGE_IPS[edge_idx], 5001, {
            'cmd': 'start_fl', 
            'round': round_num,
            'fl_round': fl_round
        })
    
    # Wait for FL completion
    print("⏳ Waiting for FL training completion...")
    edge_training_done.clear()
    timeout = time.time() + 60
    while len(edge_training_done) < len(unique_edges) and time.time() < timeout:
        time.sleep(3)
        for edge_idx in unique_edges:
            if edge_idx in edge_training_done:
                continue
            resp = send_to_socket(EDGE_IPS[edge_idx], 5001, {'cmd': 'check_status'})
            if resp and resp.get('status') == 'done':
                edge_training_done[edge_idx] = True
                print(f"  ✅ Edge {edge_idx} FL training complete")
    
    # Collect and aggregate FL results
    print("📥 Collecting FL results from edges...")
    new_weights, accuracy, loss = collect_fl_results(unique_edges)
    if new_weights is not None:
        global_model_weights = new_weights
        print("✅ Global model updated with Federated Averaging")
        # Save the updated model
        save_global_model(fl_round, accuracy, loss)
    
    # Calculate reward
    reward = calculate_reward(assignments, state)
    
    # Store experience and train DDQN
    next_state = get_state()
    done = (round_num >= ROUNDS)
    avg_action = int(np.mean([a for a in assignments if a != -1])) if any(a != -1 for a in assignments) else -1
    
    if round_num > WARMUP_ROUNDS:
        memory.append(Transition(state, avg_action, reward, next_state, float(done)))
        train_ddqn()
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    
    if round_num % TARGET_UPDATE == 0 and round_num > WARMUP_ROUNDS:
        target_net.load_state_dict(policy_net.state_dict())
        print("🔄 Target network updated")
    
    round_in_progress.clear()
    print(f"✅ Round {round_num} completed successfully")
    time.sleep(2)
    
    return state, avg_action, reward, next_state, done

def handle_device_signal(client_sock, addr):
    try:
        data = recv_msg(client_sock)
        if data and data.get('cmd') == 'ready':
            device_ready[addr[0]] = True
            device_id = data.get('device_id', 'unknown')
            print(f"📱 Device {addr[0]} (ID:{device_id}) ready for round {data.get('round', '?')}")
            send_msg(client_sock, {'status': 'acknowledged'})
        elif data and data.get('cmd') == 'data_offloaded':
            print(f"✅ Device {addr[0]} completed data offloading")
            send_msg(client_sock, {'status': 'acknowledged'})
    except Exception as e:
        print(f"Device signal error: {e}")
    finally:
        try:
            client_sock.close()
        except:
            pass

def start_server():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((CLOUD_IP, CLOUD_PORT))
    server.listen(10)
    print(f"☁️  CloudCore listening on {CLOUD_IP}:{CLOUD_PORT}")
    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_device_signal, args=(client, addr), daemon=True).start()

if __name__ == '__main__':
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(2)
    
    print("🚀 Federated Learning with Intelligent Offloading System")
    print("📊 Features: Real FL Training + Random Stress Scenarios + DDQN+ILP")
    print(f"🔧 Configuration: {NUM_DEVICES} devices, {NUM_EDGES} edges, {ROUNDS} rounds")
    print(f"🎯 Stress Levels: NO_STRESS, LOW_STRESS, MED_STRESS, HIGH_STRESS")
    print(f"💾 Model Saving: Enabled (saved_models/ directory)")
    
    # Track results
    results = {
        'rewards': [], 'assignments': [], 
        'stress_levels': [], 'fl_accuracy': [], 'fl_loss': []
    }
    
    successful_rounds = 0
    
    for r in range(1, ROUNDS + 1):
        try:
            result = run_round(r)
            if result[0] is not None:
                state, action, reward, next_state, done = result
                results['rewards'].append(reward)
                results['assignments'].append(action)
                results['stress_levels'].append(edge_stress_levels.copy())
                if fl_accuracy_history:
                    results['fl_accuracy'].append(fl_accuracy_history[-1])
                    results['fl_loss'].append(fl_loss_history[-1])
                successful_rounds += 1
        except Exception as e:
            print(f"❌ Error in round {r}: {e}")
            continue
    
    # Final results
    print(f"\n{'='*60}")
    print("🎉 EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"📈 FINAL RESULTS:")
    print(f"   Successful Rounds: {successful_rounds}/{ROUNDS}")
    print(f"   Average Reward: {np.mean(results['rewards']):.3f}")
    
    if results['fl_accuracy']:
        final_accuracy = results['fl_accuracy'][-1]
        best_accuracy = max(results['fl_accuracy'])
        print(f"   Final FL Accuracy: {final_accuracy:.4f}")
        print(f"   Best FL Accuracy: {best_accuracy:.4f}")
        print(f"   Final FL Loss: {results['fl_loss'][-1]:.4f}")
    
    # Stress level distribution
    stress_counts = {}
    for stress_levels in results['stress_levels']:
        for edge, level in stress_levels.items():
            stress_counts[level] = stress_counts.get(level, 0) + 1
    
    print(f"\n📊 STRESS DISTRIBUTION:")
    for level, count in stress_counts.items():
        percentage = (count / (len(results['stress_levels']) * NUM_EDGES)) * 100
        print(f"   {level}: {count} occurrences ({percentage:.1f}%)")
    
    # Edge utilization
    edge_utilization = [0] * NUM_EDGES
    for assignment in results['assignments']:
        if isinstance(assignment, (list, tuple)):
            for edge in assignment:
                if 0 <= edge < NUM_EDGES:
                    edge_utilization[edge] += 1
        elif 0 <= assignment < NUM_EDGES:
            edge_utilization[assignment] += 1
    
    print(f"\n📊 EDGE UTILIZATION:")
    for edge_idx, count in enumerate(edge_utilization):
        utilization = (count / len(results['assignments'])) * 100
        print(f"   Edge {edge_idx}: {count} assignments ({utilization:.1f}%)")
    
    print(f"\n💾 Saved Models: check 'saved_models/' directory")
    print("✅ Experiment completed successfully!")

