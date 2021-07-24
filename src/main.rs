#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;

use crate::render::vulkan_context::VulkanContext;
use crate::display::Display;
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::ops::{Deref, DerefMut};

mod block_world;
mod display;
mod render;

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
                if display.render().expect("Rendering failed") {
                    display.device().device_wait_idle().expect("Failed to wait device idle!");
                    display.recreate().expect("Swapchain recreation failed");
                }
            }
            _ => (),
        }
    })
}