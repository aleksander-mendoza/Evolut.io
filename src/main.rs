#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;





use failure::err_msg;

// use crate::triangles::Triangles;

use pipelines::display::Display;
use pipelines::game::GameResources;



use pipelines::player::Player;



use crate::fps::FpsCounter;
use crate::input::Input;








use crate::render::vulkan_context::VulkanContext;








mod render;
mod input;
mod fps;
mod blocks;
mod pipelines;
mod physics_timer;
mod neat;


fn main() -> Result<(), failure::Error> {
    run()
}


fn run() -> Result<(),failure::Error>{
    #[cfg(target_os = "macos")] {
        sdl2::hint::set("SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK", "1");
    }
    let sdl = sdl2::init().map_err(err_msg)?;
    let video_subsystem = sdl.video().map_err(err_msg)?;
    let timer = sdl.timer().map_err(err_msg)?;
    let window = video_subsystem
        .window("∑volut-io", 900, 700)
        .vulkan()
        .resizable()
        .build()?;
    // sdl.mouse().set_relative_mouse_mode(true);
    let vulkan = VulkanContext::new(window)?;
    let mut player = Player::new();
    let mut display = Display::new(vulkan,&player, GameResources::new)?;
    let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(event_pump);
    let mut fps_counter = FpsCounter::new(timer, 60);

    player.resize(&display);
    display.rerecord_all_cmd_buffers()?;
    'main: loop {
        fps_counter.update();
        input.poll();

        if input.quit() {
            break;
        }
        if input.escape() {
            input.reset_escape();
            sdl.mouse()
                .set_relative_mouse_mode(!sdl.mouse().relative_mouse_mode());
        }
        player.update(&mut display, &input, &fps_counter);
        // player.send_events(sx);
        if display.render(false,&mut player)? {
            display.device().device_wait_idle()?;
            display.recreate()?;
            display.rerecord_all_cmd_buffers()?;
            player.resize(&display);
        }
    }
    display.device().device_wait_idle()?;
    Ok(())
}
