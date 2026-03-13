import copy
from detour_engine import calculate_walk_penalty

class ServiceSolution:
    """Represents a complete state of student-to-route assignments."""
    
    def __init__(self, students, routes, graph):
        """
        Args:
            students: List of all Student objects.
            routes: List of active Route objects.
            graph: Reference to the road network (read-only).
        """
        self.students = students
        self.routes = routes
        self.graph = graph
        
    def calculate_objective(self):
        """
        Formula: (count_served * 10000) - sum(route_travel_times) - walk_penalties
        Walk penalties discourage assigning students to stops beyond their
        recommended walking radius, while still allowing it when necessary.
        """
        # Optimized objective calculation
        served_count = 0
        total_walk_penalty = 0.0
        total_time = 0.0
        active_routes = 0
        
        from alns_engine import _blazing_fast_walk_penalty
        
        for r in self.routes:
            sc = r.get_student_count()
            if sc > 0:
                active_routes += 1
                total_time += r.total_time
                for stop in r.stops:
                    for student in stop.students:
                        served_count += 1
                        penalty, _, _ = _blazing_fast_walk_penalty(
                            student, stop.node_id, self.graph
                        )
                        if penalty == float('inf'):
                            total_walk_penalty += 5000
                        else:
                            total_walk_penalty += penalty
        
        return (served_count * 10000) - (active_routes * 5000) - total_time - total_walk_penalty
        
    def clone(self):
        """
        Creates a deep clone by rebuilding the student-stop relationships.
        Ensures the new solution's objects do not point back to the old one.
        Optimized to avoid the massive overhead of copy.copy().
        """
        from entities import Student, Stop, Route
        
        # 1. Clone students and reset temporary state
        new_students = []
        student_map = {}
        for s in self.students:
            # Manually recreate Student to avoid copy.copy overhead
            ns = Student(s.id, s.coords[0], s.coords[1], s.age, s.school_stage, s.fee,
                         s.assignment, s.valid_from, s.valid_until, 
                         walk_radius=s.walk_radius)
            ns.direct_time_to_school = s.direct_time_to_school
            ns.direct_time_from_school = s.direct_time_from_school
            # assigned_stop and is_served are reset to None/False by __init__
            new_students.append(ns)
            student_map[ns.id] = ns
            
        # 2. Clone routes and their internal stops
        new_routes = []
        for old_route in self.routes:
            # Manually recreate Route
            nr = Route(old_route.bus, old_route.route_id, old_route.route_tmax,
                       old_route.ride_time_multiplier, old_route.floor_minutes, 
                       old_route.ceiling_minutes, old_route.bidirectional_check)
            nr.total_distance = old_route.total_distance
            nr.total_time = old_time = old_route.total_time
            nr.detour_time_used = old_route.detour_time_used
            
            for old_stop in old_route.stops:
                # Manually recreate Stop
                ns = Stop(old_stop.node_id, old_stop.coords[0], old_stop.coords[1], 
                          old_stop.stop_id, old_stop.stop_type)
                
                # Re-link corresponding new students to this new stop
                for old_val_student in old_stop.students:
                    if old_val_student.id in student_map:
                        ns.add_student(student_map[old_val_student.id])
                
                nr.stops.append(ns)
            
            new_routes.append(nr)
            
        return ServiceSolution(new_students, new_routes, self.graph)

    def __repr__(self):
        served = sum(1 for s in self.students if s.is_served)
        return f"ServiceSolution(served={served}/{len(self.students)}, objective={self.calculate_objective():.2f})"
