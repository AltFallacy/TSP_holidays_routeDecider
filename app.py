from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app) 


CSV_PATH = r"C:\datasets\world\worldcities.csv" 
try:
    world_cities_df = pd.read_csv(CSV_PATH)
    world_cities_df = world_cities_df.dropna(subset=['lat', 'lng'])
except FileNotFoundError:
    print(f"FATAL ERROR: Could not find the cities file at: {CSV_PATH}")
    world_cities_df = None

selected_points = [] 



def haversine(a, b):
    R = 6371
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))

def held_karp(dist):
    n = len(dist)
    N = 1 << n
    dp = [[float('inf')] * n for _ in range(N)]
    parent = [[-1] * n for _ in range(N)]
    dp[1][0] = 0

    for mask in range(1, N):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev = mask ^ (1 << j)
            if prev == 0:
                continue
            for k in range(n):
                if not (prev & (1 << k)):
                    continue
                new_cost = dp[prev][k] + dist[k][j]
                if new_cost < dp[mask][j]:
                    dp[mask][j] = new_cost
                    parent[mask][j] = k

    full = (1 << n) - 1
    best = float('inf')
    last = -1
    for j in range(1, n):
        cost = dp[full][j] + dist[j][0]
        if cost < best:
            best = cost
            last = j

    if last == -1 and n > 0:
        last = 0 
    
    path = []
    mask = full
    
    while last != -1:
        path.append(last)
        prev_last = parent[mask][last]
        mask ^= (1 << last)
        last = prev_last

    path.reverse()
    return best, path


@app.route('/')
def index():
    return render_template('world_cities_map.html')

@app.route('/get_city_coords', methods=['GET'])
def get_city_coords():
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({'status': 'error', 'message': 'No city name provided'}), 400
    
    if world_cities_df is None:
        return jsonify({'status': 'error', 'message': 'Server error: City dataset not loaded'}), 500
        
    result = world_cities_df[world_cities_df['city'].str.lower() == city_name.lower()]
    
    if result.empty:
        result = world_cities_df[world_cities_df['city'].str.lower().str.contains(city_name.lower())]

    if result.empty:
        return jsonify({'status': 'error', 'message': f'City "{city_name}" not found'}), 404
    
    top_result = result.iloc[0]
    return jsonify({
        'status': 'ok',
        'name': top_result['city'],
        'lat': top_result['lat'],
        'lon': top_result['lng']
    })

@app.route('/add_point', methods=['POST'])
def add_point():
    global selected_points
    data = request.get_json(force=True)
    
    if len(selected_points) >= 15:
        return jsonify({'status': 'error', 'message': 'Max 15 points allowed'})
        
    point = [data['name'], data['lat'], data['lon']]
    if point not in selected_points:
        selected_points.append(point)
        
    return jsonify({'status': 'ok', 'points': selected_points})

@app.route('/clear_points', methods=['POST'])
def clear_points():
    global selected_points
    selected_points = []
    return jsonify({'status': 'ok', 'points': selected_points})

@app.route('/solve_tsp', methods=['GET'])
def solve_tsp():
    global selected_points
    if len(selected_points) < 2:
        return jsonify({'error': 'Need at least 2 points'})

    n = len(selected_points)
    dist = [[0]*n for _ in range(n)]
    coords = []
    
    for i in range(n):
        coords.append((selected_points[i][1], selected_points[i][2]))

    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine(coords[i], coords[j])
            dist[i][j] = distance
            dist[j][i] = distance

    min_cost, tour_indices = held_karp(dist)
    
    if min_cost == float('inf'):
        return jsonify({'error': 'Could not find a valid tour'})
        
    route = [selected_points[i] for i in tour_indices]

    return jsonify({'route': route, 'cost': min_cost})

if __name__ == '__main__':
    app.run(debug=True)

