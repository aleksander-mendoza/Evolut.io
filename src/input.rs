use nalgebra_glm as glm;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::error::ExternalError;
use winit::event::{ElementState, MouseButton, WindowEvent};
use crate::VulkanContext;

pub struct Input {
    quit: bool,
    escape: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    forward: bool,
    backward: bool,
    window_width: i32,
    window_height: i32,
    has_resize: bool,
    hide_mouse: bool,
    mouse_x: f64,
    mouse_y: f64,
    prev_mouse_x: f64,
    prev_mouse_y: f64,
    mouse_move_xrel: f64,
    mouse_move_yrel: f64,
    has_mouse_move: bool,
    has_mouse_left_click: bool,
    has_mouse_right_click: bool,
    has_mouse_left_down: bool,
    has_mouse_right_down: bool,
    q: bool,
    e: bool,
    pause: bool,
    next: bool,
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
    verbose: bool,
}

impl Input {
    pub fn new(window: &winit::window::Window) -> Input {
        let PhysicalSize { width, height } = window.inner_size();
        Input {
            quit: false,
            escape: false,
            left: false,
            right: false,
            up: false,
            down: false,
            forward: false,
            backward: false,
            window_width: width as i32,
            window_height: height as i32,
            has_resize: false,
            hide_mouse: false,
            mouse_x: 0.,
            mouse_y: 0.,
            prev_mouse_x: 0.,
            prev_mouse_y: 0.,
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
            pause: false,
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
            number: 0,
            next: false,
            verbose: false,
        }
    }
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose
    }
    pub fn reset(&mut self) {
        self.has_resize = false;
        self.has_mouse_move = false;
        self.has_mouse_left_click = false;
        self.has_mouse_right_click = false;
        self.q = false;
        self.pause = false;
        self.next = false;
        self.number = -1;
        self.prev_mouse_x = self.mouse_x;
        self.prev_mouse_y = self.mouse_y;
    }
    pub fn is_mouse_hidden(&self) -> bool {
        self.hide_mouse
    }
    pub fn set_mouse_hidden(&mut self, hide: bool, window: &winit::window::Window) {
        self.hide_mouse = hide;
        window.set_cursor_visible(!hide)
    }
    pub fn finish_poll(&mut self, window: &winit::window::Window) -> Result<(), ExternalError> {
        if self.hide_mouse {
            self.center_cursor(window)
        } else {
            Ok(())
        }
    }
    pub fn center_cursor(&mut self, window: &winit::window::Window) -> Result<(), ExternalError> {
        let center = PhysicalPosition::new(self.window_width / 2, self.window_height / 2);
        self.mouse_x = center.x as f64;
        self.mouse_y = center.y as f64;
        window.set_cursor_position(center)
    }

    pub fn poll(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.quit = true,
            WindowEvent::Resized(size) => {
                self.window_width = size.width as i32;
                self.window_height = size.height as i32;
                self.has_resize = true;
            }
            WindowEvent::KeyboardInput { device_id, is_synthetic, input } => {
                if self.verbose {
                    println!("Keyboard {:?}", input);
                }
                match input.state {
                    ElementState::Pressed => {
                        if let Some(k) = input.virtual_keycode {
                            match k {
                                winit::event::VirtualKeyCode::Numpad0 => {
                                    self.no0 = true;
                                    self.number = 0;
                                }
                                winit::event::VirtualKeyCode::Numpad1 => {
                                    self.no1 = true;
                                    self.number = 1;
                                }
                                winit::event::VirtualKeyCode::Numpad2 => {
                                    self.no2 = true;
                                    self.number = 2;
                                }
                                winit::event::VirtualKeyCode::Numpad3 => {
                                    self.no3 = true;
                                    self.number = 3;
                                }
                                winit::event::VirtualKeyCode::Numpad4 => {
                                    self.no4 = true;
                                    self.number = 4;
                                }
                                winit::event::VirtualKeyCode::Numpad5 => {
                                    self.no5 = true;
                                    self.number = 5;
                                }
                                winit::event::VirtualKeyCode::Numpad6 => {
                                    self.no6 = true;
                                    self.number = 6;
                                }
                                winit::event::VirtualKeyCode::Numpad7 => {
                                    self.no7 = true;
                                    self.number = 7;
                                }
                                winit::event::VirtualKeyCode::Numpad8 => {
                                    self.no8 = true;
                                    self.number = 8;
                                }
                                winit::event::VirtualKeyCode::Numpad9 => {
                                    self.no9 = true;
                                    self.number = 9;
                                }
                                winit::event::VirtualKeyCode::R => {
                                    self.r = true;
                                }
                                winit::event::VirtualKeyCode::E => {
                                    self.e = true;
                                }
                                winit::event::VirtualKeyCode::Q => {
                                    self.q = true;
                                }
                                winit::event::VirtualKeyCode::Right => {
                                    self.next = true;
                                }
                                winit::event::VirtualKeyCode::P => {
                                    self.pause = true;
                                }
                                winit::event::VirtualKeyCode::D => {
                                    self.right = true;
                                }
                                winit::event::VirtualKeyCode::A => {
                                    self.left = true;
                                }
                                winit::event::VirtualKeyCode::W => {
                                    self.forward = true;
                                }
                                winit::event::VirtualKeyCode::S => {
                                    self.backward = true;
                                }
                                winit::event::VirtualKeyCode::Space => {
                                    self.up = true;
                                }
                                winit::event::VirtualKeyCode::LShift => {
                                    self.down = true;
                                }
                                winit::event::VirtualKeyCode::Escape => {
                                    self.escape = true;
                                }
                                _ => (),
                            }
                        }
                    }
                    ElementState::Released => {
                        if let Some(k) = input.virtual_keycode {
                            match k {
                                winit::event::VirtualKeyCode::Numpad0 => {
                                    self.no0 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad1 => {
                                    self.no1 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad2 => {
                                    self.no2 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad3 => {
                                    self.no3 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad4 => {
                                    self.no4 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad5 => {
                                    self.no5 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad6 => {
                                    self.no6 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad7 => {
                                    self.no7 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad8 => {
                                    self.no8 = false;
                                }
                                winit::event::VirtualKeyCode::Numpad9 => {
                                    self.no9 = false;
                                }
                                winit::event::VirtualKeyCode::R => {
                                    self.r = false;
                                }
                                winit::event::VirtualKeyCode::E => {
                                    self.e = false;
                                }
                                winit::event::VirtualKeyCode::Q => {
                                    self.q = false;
                                }
                                winit::event::VirtualKeyCode::D => {
                                    self.right = false;
                                }
                                winit::event::VirtualKeyCode::A => {
                                    self.left = false;
                                }
                                winit::event::VirtualKeyCode::W => {
                                    self.forward = false;
                                }
                                winit::event::VirtualKeyCode::S => {
                                    self.backward = false;
                                }
                                winit::event::VirtualKeyCode::Space => {
                                    self.up = false;
                                }
                                winit::event::VirtualKeyCode::LShift => {
                                    self.down = false;
                                }
                                winit::event::VirtualKeyCode::Escape => {
                                    self.escape = false;
                                }
                                _ => (),
                            }
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x;
                self.mouse_y = position.y;
                self.mouse_move_xrel = self.mouse_x - self.prev_mouse_x;
                self.mouse_move_yrel = self.mouse_y - self.prev_mouse_y;
                self.has_mouse_move = true;
                if self.verbose {
                    println!("Mouse {{position:{{x:{},y:{}}},moved:{{x:{},y:{}}}}}", self.mouse_x, self.mouse_y, self.mouse_move_xrel, self.mouse_move_yrel);
                }
            }
            // Accumulate input events
            WindowEvent::MouseInput {
                button,
                state,
                ..
            } => {
                if self.verbose{
                    println!("Mouse button {:?}, {:?}", button, state)
                }
                match state {
                    ElementState::Pressed => match button {
                        MouseButton::Left => {
                            if !self.has_mouse_left_down {
                                self.has_mouse_left_click = true;
                            }
                            self.has_mouse_left_down = true;
                        }
                        MouseButton::Right => {
                            if !self.has_mouse_right_down {
                                self.has_mouse_right_click = true;
                            }
                            self.has_mouse_right_down = true;
                        }
                        _ => {}
                    }
                    ElementState::Released => match button {
                        MouseButton::Left => {
                            self.has_mouse_left_down = false;
                        }
                        MouseButton::Right => {
                            self.has_mouse_right_down = false;
                        }
                        _ => {}
                    }
                }
            }
            _ => (),
        }
    }
    pub fn has_resize(&self) -> bool {
        self.has_resize
    }
    pub fn resize_w(&self) -> i32 {
        self.window_width
    }
    pub fn resize_h(&self) -> i32 {
        self.window_height
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
    pub fn mouse_x(&self) -> f64 {
        self.mouse_x
    }
    pub fn mouse_y(&self) -> f64 {
        self.mouse_y
    }
    pub fn mouse_move_xrel(&self) -> f64 {
        self.mouse_move_xrel
    }
    pub fn mouse_move_yrel(&self) -> f64 {
        self.mouse_move_yrel
    }
    pub fn quit(&self) -> bool {
        self.quit
    }
    pub fn pause(&self) -> bool {
        self.pause
    }
    pub fn next(&self) -> bool {
        self.next
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
