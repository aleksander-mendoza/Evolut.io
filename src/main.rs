#[macro_use]
extern crate vk_shader_macros;

use crate::vulkan_context::VulkanContext;
use crate::display::Display;
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::ops::{Deref, DerefMut};

mod window;
mod instance;
mod constants;
mod platforms;
mod validation_layer;
mod device;
mod surface;
mod swap_chain;
mod block_world;
mod shader_module;
mod imageview;
mod pipeline;
mod render_pass;
mod command_pool;
mod semaphore;
mod fence;
mod frames_in_flight;
mod vulkan_context;
mod display;


fn main() -> Result<(), failure::Error> {
    let event_loop = EventLoop::new();
    let vulkan = VulkanContext::new(&event_loop)?;
    let mut display = Display::new(vulkan)?;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode, state, .. }, .. }, .. } => {}
            Event::LoopDestroyed => {
                unsafe {
                    display.device().device_wait_idle().expect("Failed to wait device idle!");
                }
            }
            Event::MainEventsCleared => {
                if display.render().expect("Rendering failed"){
                    display.device().device_wait_idle().expect("Failed to wait device idle!");
                    display.recreate().expect("Swapchain recreation failed");
                }
            }
            _ => (),
        }
    })
}