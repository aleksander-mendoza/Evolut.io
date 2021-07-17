use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::Window;
use winit::error::OsError;
use crate::constants::APP_TITLE;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub fn init_window(event_loop: &EventLoop<()>) -> Result<Window, OsError> {
    winit::window::WindowBuilder::new()
        .with_title(APP_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)
}