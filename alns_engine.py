import random
import math
import time
import numpy as np
import copy
import networkx as nx
from detour_engine import (
    calculate_route_time_from_matrix,
    calculate_route_distance_from_matrix,
    snap_address_to_edge,
    calculate_walk_penalty,
    get_walk_absolute_max,
    _MATRIX_CACHE,
    _get_walk_graph,
    find_safe_nodes_within_radius
)
from entities import Stop

_student_candidate_cache = {}
_student_candidate_dist = {}
_fast_walk_cache = {}

def _blazing_fast_walk_penalty(student, node_id, graph, unconstrained=False):
    key = (student.id, node_id, unconstrained)
    if key not in _fast_walk_cache:
        p, m, over = calculate_walk_penalty(student, node_id, graph)
        if unconstrained and p == float('inf') and m <= get_walk_absolute_max(student.walk_radius):
            p = 0.0 
        _fast_walk_cache[key] = (p, m, over)
    return _fast_walk_cache[key]

# ============================================================================
# DESTROY OPERATORS
# ============================================================================

def random_removal(solution, n):
    served = [s for s in solution.students if s.is_served]
    n = min(n, len(served))
    if n == 0: return []
    removed = random.sample(served, n)
    for s in removed: _remove_student(solution, s)
    return removed

def worst_cost_removal(solution, n):
    served = [s for s in solution.students if s.is_served]
    n = min(n, len(served))
    if n == 0: return []
    costs = []
    for s in served:
        stop = s.assigned_stop
        route = next((r for r in solution.routes if stop in r.stops), None)
        if not route: continue
        old_t = route.total_time
        # Time of route without this student
        temp_stops = [st for st in route.stops if st != stop or len(st.students) > 1]
        new_t = calculate_route_time_from_matrix(temp_stops, solution.graph)
        costs.append((s, old_t - (new_t if new_t is not None else old_t)))
    costs.sort(key=lambda x: x[1], reverse=True)
    removed = [c[0] for c in costs[:n]]
    for s in removed: _remove_student(solution, s)
    return removed

def _remove_student(solution, student):
    stop = student.assigned_stop
    if not stop: return
    route = next((r for r in solution.routes if stop in r.stops), None)
    if not route: return
    stop.remove_student(student)
    if not stop.students: route.stops.remove(stop)
    t = calculate_route_time_from_matrix(route.stops, solution.graph)
    route.total_time = t if t is not None else 0.0

# ============================================================================
# REPAIR OPERATORS
# ============================================================================

def regret_repair(solution, k=2, unconstrained=False):
    unassigned = [s for s in solution.students if not s.is_served]
    if not unassigned: return
    
    # Pre-calculate options for all unassigned
    options_map = {s.id: {r.route_id: _get_insertions(s, r, solution.graph, unconstrained) 
                   for r in solution.routes} for s in unassigned}

    while unassigned:
        best_regret, target_s, target_ins = -1, None, None
        for s in unassigned:
            all_opts = []
            for r_opts in options_map[s.id].values(): all_opts.extend(r_opts)
            if not all_opts: continue
            all_opts.sort(key=lambda x: x['cost'])
            
            regret = (all_opts[k-1]['cost'] - all_opts[0]['cost']) if len(all_opts) >= k else (2000 - all_opts[0]['cost'])
            if regret > best_regret:
                best_regret, target_s, target_ins = regret, s, all_opts[0]
        
        if target_s and target_ins:
            route = target_ins['route']
            _apply_ins(solution, target_s, target_ins)
            unassigned.remove(target_s)
            # Update only affected route for remaining students
            for s in unassigned: 
                options_map[s.id][route.route_id] = _get_insertions(s, route, solution.graph, unconstrained)
        else: break

def _get_insertions(student, route, graph, unconstrained=False):
    if route.get_student_count() >= route.bus.capacity: return []
    candidates = _student_candidate_cache.get(student.id, [])
    if not candidates: return []

    results = []
    n_stops = len(route.stops)
    if n_stops < 2: return []

    k_r, fl_r, ce_r = getattr(route, 'ride_time_multiplier', 2.5), getattr(route, 'floor_minutes', 45), getattr(route, 'ceiling_minutes', 60)
    bidir, school_node = getattr(route, 'bidirectional_check', True), route.stops[-1].node_id

    # Precompute AM/PM baseline times
    am_times = [0.0] * n_stops
    curr_am = 0.0
    for i in range(n_stops - 1, -1, -1):
        am_times[i] = curr_am
        if i > 0: curr_am += _MATRIX_CACHE.get((route.stops[i-1].node_id, route.stops[i].node_id), 9999.0)

    pm_times = [0.0] * n_stops
    curr_pm = 0.0
    pm_order = [route.stops[-1]] + route.stops[1:-1][::-1] + [route.stops[0]]
    for i in range(n_stops):
        pm_times[i] = curr_pm
        if i < n_stops - 1: curr_pm += _MATRIX_CACHE.get((pm_order[i].node_id, pm_order[i+1].node_id), 9999.0)

    from detour_engine import compute_direct_time
    t_d = compute_direct_time(student, school_node, graph)
    cap = max(fl_r, min(k_r * t_d, t_d + ce_r)) if t_d < 9999 else 9999.0

    # Pre-calculate caps for existing students on this route to avoid repeated lookups
    existing_student_caps = []
    for i, stop in enumerate(route.stops):
        if stop.stop_type == 'school': continue
        for st in stop.students:
            # We need the direct time to calculate their dynamic cap
            t_direct = compute_direct_time(st, school_node, graph)
            st_cap = max(fl_r, min(k_r * t_direct, t_direct + ce_r)) if t_direct < 9999 else 9999.0
            # (boarding_position, current_ride_time, capacity)
            existing_student_caps.append((i, am_times[i], st_cap))

    for pos in range(1, n_stops):
        u, v = route.stops[pos-1].node_id, route.stops[pos].node_id
        dt_old = _MATRIX_CACHE.get((u, v), 9999.0)
        for nid, coords in candidates:
            dt1, dt2 = _MATRIX_CACHE.get((u, nid), 9999.0), _MATRIX_CACHE.get((nid, v), 9999.0)
            if dt1 >= 9999 or dt2 >= 9999: continue
            
            delta = dt1 + dt2 - dt_old
            
            # 1. NEW student comfort check
            if (dt2 + am_times[pos]) > cap:
                if not bidir: continue
                pm_idx_v = n_stops - 1 - pos
                pm_v_to_new = _MATRIX_CACHE.get((v, nid), 9999.0)
                if (pm_times[pm_idx_v] + pm_v_to_new) > cap: continue

            # 2. EXISTING students comfort check (The Domino Effect Fix)
            too_long = False
            for board_pos, current_time, st_cap in existing_student_caps:
                if board_pos < pos: # These students are delayed by the new stop
                    if (current_time + delta) > st_cap:
                        too_long = True
                        break
            if too_long: continue

            w_pen, _, _ = _blazing_fast_walk_penalty(student, nid, graph, unconstrained)
            if w_pen == float('inf'): continue

            existing = next((s for s in route.stops if s.node_id == nid), None)
            results.append({'route': route, 'cost': delta + w_pen, 'pos': pos,
                'stop': existing if existing else Stop(nid, coords[0], coords[1]), 'is_new': existing is None})
    return results

def _apply_ins(sol, student, ins):
    route, stop = ins['route'], ins['stop']
    if ins['is_new']: route.stops.insert(ins['pos'], stop)
    stop.add_student(student)
    t = calculate_route_time_from_matrix(route.stops, sol.graph)
    route.total_time = t if t is not None else 0.0

# ============================================================================
# ALNS ENGINE
# ============================================================================

class ALNSEngine:
    def __init__(self, initial_sol, iterations=100, time_budget_seconds=None, max_candidates_per_student=3, unconstrained=False):
        self.curr_sol = initial_sol.clone()
        self.best_sol = initial_sol.clone()
        self.iterations = iterations
        self.time_budget_seconds = time_budget_seconds
        self.unconstrained = unconstrained
        self.destroy_ops = [random_removal, worst_cost_removal]
        
        mode_label = "DOOR-TO-DOOR" if all(s.walk_radius == 0 for s in self.curr_sol.students) else ("UNCONSTRAINED" if unconstrained else "SAFE")
        print(f"  Pre-calculating {mode_label} menus for {len(self.curr_sol.students)} students...")
        total_options = 0
        for s in self.curr_sol.students:
            nid, coords = snap_address_to_edge(s.coords, initial_sol.graph)
            menu = [(nid, coords)]
            if s.walk_radius > 0:
                if unconstrained:
                    walk_g = _get_walk_graph(initial_sol.graph)
                    lengths = nx.single_source_dijkstra_path_length(walk_g, nid, cutoff=s.walk_radius, weight='length')
                    for snid in sorted(lengths, key=lengths.get)[1:max_candidates_per_student]:
                        node = initial_sol.graph.nodes[snid]
                        menu.append((snid, (node['y'], node['x'])))
                else:
                    safe = find_safe_nodes_within_radius(s.coords, initial_sol.graph, 500, s.walk_radius, {"max_candidates_per_student": max_candidates_per_student})
                    for snid, _ in safe:
                        if snid != nid:
                            node = initial_sol.graph.nodes[snid]
                            menu.append((snid, (node['y'], node['x'])))
            _student_candidate_cache[s.id] = menu[:max_candidates_per_student]
            total_options += len(_student_candidate_cache[s.id])
        print(f"  Pre-calculation complete. Avg options/student: {total_options/len(self.curr_sol.students):.1f}")

    def run(self):
        start = time.time()
        print(f"  Optimization starting ({self.iterations} iters)...")
        for i in range(self.iterations):
            if self.time_budget_seconds and (time.time() - start) >= self.time_budget_seconds: break
            
            new_sol = self.curr_sol.clone()
            # 1. Destroy: remove 15-25% of students
            n_rem = max(1, int(len(new_sol.students) * random.uniform(0.15, 0.25)))
            random.choice(self.destroy_ops)(new_sol, n_rem)
            
            # 2. Repair
            regret_repair(new_sol, unconstrained=self.unconstrained)
            
            # 3. Objective check
            new_obj = new_sol.calculate_objective()
            best_obj = self.best_sol.calculate_objective()
            
            if new_obj > best_obj:
                self.best_sol = new_sol.clone()
                self.curr_sol = new_sol
            elif random.random() < 0.1: # Acceptance criteria (SA)
                self.curr_sol = new_sol
            
            if (i+1) % 10 == 0:
                print(f"    Iteration {i+1}: Best Obj = {self.best_sol.calculate_objective():.0f}")

        return self.best_sol
