use std::time::{Duration, Instant};

pub struct FpsCounter {
    previous: Instant,
    delta: u32,
    frames_per_second: u32,
}

impl FpsCounter {
    pub fn new(frames_per_second:u32) -> Self {
        let previous = Instant::now();
        Self {
            previous,
            delta: 0,
            frames_per_second
        }
    }
    pub fn update(&mut self) {
        let delta = self.previous.elapsed().as_millis() as u64;
        let proportion = delta*self.frames_per_second as u64;
        // delta * frames_per_second < 1000
        // delta < 1000/frames_per_second
        // delta < millis_per_frame
        if proportion < 1000 {
            // (1000 - delta * frames_per_second)/frames_per_second
            // == 1000/frames_per_second - delta
            // == millis_per_frame - delta
            let sleep_time = (1000u64 - proportion) / self.frames_per_second as u64;
            std::thread::sleep(Duration::from_millis(sleep_time));
        }
        self.delta = self.previous.elapsed().as_millis() as u32;
        self.previous = Instant::now();
    }
    pub fn delta(&self) -> u32 {
        self.delta
    }
    pub fn delta_f32(&self) -> f32 {
        self.delta as f32
    }
    pub fn ticks(&self) -> Instant {
        self.previous
    }
}
