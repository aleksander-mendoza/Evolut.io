mod window;
mod instance;
mod constants;
mod platforms;
mod validation_layer;
mod debug;

use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::Window;
use winit::error::OsError;
use crate::window::init_window;
use crate::instance::Instance;


struct VulkanApp;

impl VulkanApp {

    pub fn main_loop(event_loop: EventLoop<()>) {

        event_loop.run(move |event, _, control_flow| {

            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | KeyboardInput { virtual_keycode, state, .. } => {
                                    match (virtual_keycode, state) {
                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                            dbg!();
                                            *control_flow = ControlFlow::Exit
                                        },
                                        | _ => {},
                                    }
                                },
                            }
                        },
                        | _ => {},
                    }
                },
                _ => (),
            }

        })
    }
}

fn main() -> Result<(), failure::Error>{
    let entry = unsafe{ash::Entry::new()}?;
    let event_loop = EventLoop::new();
    let window = init_window(&event_loop)?;
    let instance = Instance::new(&entry, true)?;
    VulkanApp::main_loop(event_loop);
    Ok(())
}