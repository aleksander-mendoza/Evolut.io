
pub fn ray_cast<T, F: FnMut(f32, f32, f32, f32, f32, f32) -> Option<T>>(start: &[f32], distance_and_direction: &[f32], mut f: F) -> Option<T> {
    //initial point A
    let (ax, ay, az) = (start[0], start[1], start[2]);
    //distance vector D
    let (dx, dy, dz) = (distance_and_direction[0], distance_and_direction[1], distance_and_direction[2]);
    //current voxel boundary
    let (mut vx, mut vy, mut vz) = (ax.floor(), ay.floor(), az.floor());
    let o = f(vx, vy, vz, vx, vy, vz);
    if o.is_some() {
        return o;
    }
    //final voxel boundary B
    let (bx, by, bz) = (ax + dx, ay + dy, az + dz);
    let bv = (bx.floor(), by.floor(), bz.floor());
    fn compute_step_and_initial_ray_length(d: f32, a: f32, v: f32) -> (f32, f32) {
        if d < 0. {
            (-1f32, (v - a) / d)//notice that the signs will cancel out and the result will be positive
        } else {
            (1f32, (1f32 + v - a) / d)
        }
    }
    let (step_x, mut t_max_x) = compute_step_and_initial_ray_length(dx, ax, vx);
    let (step_y, mut t_max_y) = compute_step_and_initial_ray_length(dy, ay, vy);
    let (step_z, mut t_max_z) = compute_step_and_initial_ray_length(dz, az, vz);
    let t_delta_x = step_x / dx;//notice that the signs will cancel out. Division by zero will yield +inf
    assert!(t_delta_x >= 0f32);
    let t_delta_y = step_y / dy;
    let t_delta_z = step_z / dz;

    while (vx, vy, vz) != bv {
        let o = if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                let new_vx = vx + step_x;
                let o = f(new_vx, vy, vz, vx, vy, vz);
                vx = new_vx;
                t_max_x += t_delta_x;
                o
            } else {
                let new_vz = vz + step_z;
                let o = f(vx, vy, new_vz, vx, vy, vz);
                vz = new_vz;
                t_max_z += t_delta_z;
                o
            }
        } else {
            if t_max_y < t_max_z {
                let new_vy = vy + step_y;
                let o = f(vx, new_vy, vz, vx, vy, vz);
                vy = new_vy;
                t_max_y += t_delta_y;
                o
            } else {
                let new_vz = vz + step_z;
                let o = f(vx, vy, new_vz, vx, vy, vz);
                vz = new_vz;
                t_max_z += t_delta_z;
                o
            }
        };

        if o.is_some() {
            return o;
        }
    }
    None
}