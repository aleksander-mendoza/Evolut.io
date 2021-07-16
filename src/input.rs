use nalgebra_glm as glm;
use winit::event::{ElementState, VirtualKeyCode, MouseButton};
use winit::dpi::PhysicalPosition;

pub struct Input {
    escape: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    forward: bool,
    backward: bool,
    mouse_move_x: f64,
    mouse_move_y: f64,
    mouse_move_xrel: f64,
    mouse_move_yrel: f64,
    has_mouse_move: bool,
    has_mouse_left_click: bool,
    has_mouse_right_click: bool,
    has_mouse_left_down: bool,
    has_mouse_right_down: bool,
    q: bool,
    e: bool,
    r: bool,
    no0: bool,
    no1: bool,
    no2: bool,
    no3: bool,
    no4: bool,
    no5: bool,
    no6: bool,
    no7: bool,
    no8: bool,
    no9: bool,
    number: i32,
}

impl Input {
    pub fn new() -> Input {
        Input {
            escape: false,
            left: false,
            right: false,
            up: false,
            down: false,
            forward: false,
            backward: false,
            mouse_move_x: 0.,
            mouse_move_y: 0.,
            mouse_move_xrel: 0.,
            mouse_move_yrel: 0.,
            has_mouse_move: false,
            has_mouse_left_click: false,
            has_mouse_right_click: false,
            has_mouse_left_down: false,
            has_mouse_right_down: false,
            q: false,
            e: false,
            r: false,
            no0: false,
            no1: false,
            no2: false,
            no3: false,
            no4: false,
            no5: false,
            no6: false,
            no7: false,
            no8: false,
            no9: false,
            number: 0
        }
    }
    pub fn update_keyboard(&mut self, state:ElementState, key:VirtualKeyCode) {
        let pressed = match state {
            ElementState::Pressed => true,
            ElementState::Released => false
        };
        match key {
            VirtualKeyCode::Key0 => {
                self.no0 = pressed;
                self.number = if pressed { 0 } else { -1 };
            }
            VirtualKeyCode::Key1 => {
                self.no1 = pressed;
                self.number = if pressed { 1 } else { -1 };
            }
            VirtualKeyCode::Key2 => {
                self.no2 = pressed;
                self.number = if pressed { 2 } else { -1 };
            }
            VirtualKeyCode::Key3 => {
                self.no3 = pressed;
                self.number = if pressed { 3 } else { -1 };
            }
            VirtualKeyCode::Key4 => {
                self.no4 = pressed;
                self.number = if pressed { 4 } else { -1 };
            }
            VirtualKeyCode::Key5 => {
                self.no5 = pressed;
                self.number = if pressed { 5 } else { -1 };
            }
            VirtualKeyCode::Key6 => {
                self.no6 = pressed;
                self.number = if pressed { 6 } else { -1 };
            }
            VirtualKeyCode::Key7 => {
                self.no7 = pressed;
                self.number = if pressed { 7 } else { -1 };
            }
            VirtualKeyCode::Key8 => {
                self.no8 = pressed;
                self.number = if pressed { 8 } else { -1 };
            }
            VirtualKeyCode::Key9 => {
                self.no9 = pressed;
                self.number = if pressed { 9 } else { -1 };
            }
            VirtualKeyCode::R => {
                self.r = pressed;
            }
            VirtualKeyCode::E => {
                self.e = pressed;
            }
            VirtualKeyCode::Q => {
                self.q = pressed;
            }
            VirtualKeyCode::D => {
                self.right = pressed;
            }
            VirtualKeyCode::A => {
                self.left = pressed;
            }
            VirtualKeyCode::W => {
                self.forward = pressed;
            }
            VirtualKeyCode::S => {
                self.backward = pressed;
            }
            VirtualKeyCode::Space => {
                self.up = pressed;
            }
            VirtualKeyCode::LShift => {
                self.down = pressed;
            }
            VirtualKeyCode::Escape => {
                self.escape = pressed;
            }
            _ => {}
        }
    }
    pub fn update_mouse_position(&mut self, position:PhysicalPosition<f64>){
        self.mouse_move_xrel = position.x - self.mouse_move_x;
        self.mouse_move_yrel = position.y - self.mouse_move_y;
        self.mouse_move_x = position.x;
        self.mouse_move_y = position.y;
        self.has_mouse_move = true;
    }
    pub fn reset(&mut self){
        self.has_mouse_move = false;
        self.has_mouse_left_click = false;
        self.has_mouse_right_click = false;
        self.number = -1;
    }
    pub fn update_mouse_click(&mut self,state:ElementState,button:MouseButton){
        let pressed = match state {
            ElementState::Pressed => true,
            ElementState::Released => false
        };
        match button {
            MouseButton::Left => {
                if !self.has_mouse_left_down {
                    self.has_mouse_left_click = pressed;
                }
                self.has_mouse_left_down = pressed;
            }
            MouseButton::Right => {
                if !self.has_mouse_right_down {
                    self.has_mouse_right_click = pressed;
                }
                self.has_mouse_right_down = pressed;
            }
            _ => {}
        }
    }

    pub fn has_mouse_move(&self) -> bool {
        self.has_mouse_move
    }
    pub fn has_mouse_left_click(&self) -> bool {
        self.has_mouse_left_click
    }
    pub fn has_mouse_right_click(&self) -> bool {
        self.has_mouse_right_click
    }
    pub fn has_mouse_left_down(&self) -> bool {
        self.has_mouse_left_down
    }
    pub fn has_mouse_right_down(&self) -> bool {
        self.has_mouse_right_down
    }
    pub fn mouse_move_x(&self) -> f64 {
        self.mouse_move_x
    }
    pub fn mouse_move_y(&self) -> f64 {
        self.mouse_move_y
    }
    pub fn mouse_move_xrel(&self) -> f64 {
        self.mouse_move_xrel
    }
    pub fn mouse_move_yrel(&self) -> f64 {
        self.mouse_move_yrel
    }
    pub fn get_direction_unit_vector(&self) -> glm::TVec3<f32> {
        let x_axis = -(self.left as i32) + (self.right as i32);
        let y_axis = -(self.down as i32) + (self.up as i32);
        let z_axis = -(self.forward as i32) + (self.backward as i32);
        let length = ((x_axis * x_axis + y_axis * y_axis + z_axis * z_axis) as f32).sqrt();
        if length == 0f32 {
            return glm::vec3(0f32, 0f32, 0f32);
        }
        //normalized values:
        let x_axis = x_axis as f32 / length;
        let y_axis = y_axis as f32 / length;
        let z_axis = z_axis as f32 / length;
        glm::vec3(x_axis, y_axis, z_axis)
    }
    pub fn escape(&self) -> bool {
        self.escape
    }
    pub fn reset_escape(&mut self) {
        self.escape = false;
    }

    pub fn is_q(&self) -> bool {
        self.q
    }
    pub fn is_e(&self) -> bool {
        self.e
    }
    pub fn is_r(&self) -> bool {
        self.r
    }
    pub fn is_1(&self) -> bool {
        self.no1
    }
    pub fn is_2(&self) -> bool {
        self.no2
    }
    pub fn is_3(&self) -> bool {
        self.no3
    }
    pub fn is_4(&self) -> bool {
        self.no4
    }
    pub fn is_5(&self) -> bool {
        self.no5
    }
    pub fn is_6(&self) -> bool {
        self.no6
    }
    pub fn is_7(&self) -> bool {
        self.no7
    }
    pub fn is_8(&self) -> bool {
        self.no8
    }
    pub fn is_9(&self) -> bool {
        self.no9
    }
    pub fn number(&self) -> i32 {
        self.number
    }
}
