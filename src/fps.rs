use sdl2;
use std::time::Duration;

pub struct FpsCounter {
    timer: sdl2::TimerSubsystem,
    previous: u32,
    delta: u32,
    frames_per_second: u32,
}

impl FpsCounter {
    pub fn new(timer: sdl2::TimerSubsystem, frames_per_second:u32) -> Self {
        let previous = timer.ticks();
        Self {
            timer,
            previous,
            delta: 0,
            frames_per_second
        }
    }
    pub fn update(&mut self) {
        let current = self.timer.ticks();
        let delta = current - self.previous;
        let proportion = delta*self.frames_per_second;
        let current = if proportion < 1000 {
            let sleep_time = (1000u32 - proportion) / self.frames_per_second;
            std::thread::sleep(Duration::from_millis(sleep_time as u64));
            self.timer.ticks()
        }else{
            current
        };
        self.delta = current - self.previous;
        self.previous = current;
    }
    pub fn delta(&self) -> u32 {
        self.delta
    }
    pub fn delta_f32(&self) -> f32 {
        self.delta as f32
    }
    pub fn ticks(&self) -> u32{
        self.previous
    }
}
