#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;


use ash::vk;
use failure::err_msg;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

// use crate::triangles::Triangles;

use pipelines::display::Display;
use pipelines::game::GameResources;


use pipelines::player::Player;


use crate::fps::FpsCounter;
use crate::input::Input;


use crate::render::vulkan_context::VulkanContext;
use crate::pipelines::physics::PhysicsResources;
use crate::pipelines::ambience::AmbienceResources;


mod render;
mod input;
mod fps;
mod blocks;
mod pipelines;
mod physics_timer;
mod neat;

use winit::platform::windows::WindowExtWindows;


fn main() -> Result<(), failure::Error> {
    run()
}


fn run() -> Result<(), failure::Error> {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("∑volut-io")
        .with_inner_size(PhysicalSize::new(900, 700))
        .with_resizable(true)
        .build(&event_loop).map_err(err_msg)?;
    // let sdl = sdl2::init().map_err(err_msg)?;
    // let video_subsystem = sdl.video().map_err(err_msg)?;
    // let timer = sdl.timer().map_err(err_msg)?;
    // let window = video_subsystem
    //     .window("∑volut-io", 900, 700)
    //     .vulkan()
    //     .resizable()
    //     .build()?;
    // sdl.mouse().set_relative_mouse_mode(true);
    let vulkan = VulkanContext::new(window)?;
    let mut player = Player::new();
    let mut display = Display::new(vulkan, &player, GameResources::new, PhysicsResources::new, AmbienceResources::new)?;
    // let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(display.window());
    let mut fps_counter = FpsCounter::new(60);
    input.set_verbose(true);
    let mut run_simulation = false;
    player.resize(&display);
    display.rerecord_all_graphics_cmd_buffers()?;
    display.record_compute_cmd_buffer()?;
    display.record_background_compute_cmd_buffer()?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                // reset input states on new frame
                input.reset()
            }
            Event::MainEventsCleared => {
                // update input state after accumulating event
                fps_counter.update();
                input.finish_poll(display.window()).unwrap();
                if input.quit() {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                if input.escape() {
                    let h = !input.is_mouse_hidden();
                    println!("Cursor grabbed: {}", h);
                    input.set_mouse_hidden(h, display.window());
                    input.reset_escape();
                }
                if input.pause() {
                    run_simulation = !run_simulation;
                }

                player.update(&mut display, &input, &fps_counter);
                // player.send_events(sx);
                if run_simulation || input.next() {
                    display.compute(&mut player).unwrap();
                }

                // render

                if display.render(false, &mut player).unwrap() {
                    display.device().device_wait_idle().unwrap();
                    display.recreate_graphics().unwrap();
                    display.rerecord_all_graphics_cmd_buffers().unwrap();
                    player.resize(&display);
                }
            }
            Event::WindowEvent { event, .. } => input.poll(event),
            Event::LoopDestroyed => {
                display.device().device_wait_idle().unwrap()
            }
            _ => (),
        }
    });
}
