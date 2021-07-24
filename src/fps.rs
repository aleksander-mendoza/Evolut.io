use std::thread;
use std::time::Duration;
use std::time::Instant;

const SAMPLE_COUNT: usize = 5;

pub struct FpsCounter {
    previous: Instant,
    delta: u32,
    ticks: u32,
    frames_per_second: u32,
}

impl FpsCounter {
    pub fn new(frames_per_second:u32) -> Self {
        let previous = Instant::now();
        Self {
            previous,
            ticks: 0,
            delta: 0,
            frames_per_second
        }
    }

    pub fn update(&mut self) {
        let current = Instant::now();
        let delta = current - self.previous;
        let proportion = delta.as_millis() as u32 * self.frames_per_second;
        let current = if proportion < 1000 {
            let sleep_time = (1000u32 - proportion) / self.frames_per_second;
            std::thread::sleep(Duration::from_millis(sleep_time as u64));
            Instant::now()
        }else{
            current
        };
        self.delta = (current - self.previous).as_millis() as u32;
        self.previous = current;
        self.ticks += 1;
    }

    pub fn delta(&self) -> u32 {
        self.delta
    }
    pub fn delta_f32(&self) -> f32 {
        self.delta as f32
    }
    pub fn ticks(&self) -> u32{
        self.ticks
    }
}