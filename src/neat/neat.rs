use rand::Rng;
use rand::rngs::StdRng;
use super::num::Num;
use super::activations;
use super::cppn::CPPN;
use super::util::RandRange;
use rand::distributions::{Standard, Distribution};
use std::iter::FromIterator;


pub struct Neat<X: Num> {
    global_innovation_no: usize,
    activations: Vec<fn(X)->X>,
    input_size: usize,
    output_size: usize,
}

impl <X: Num> Neat<X> {
    pub fn get_activation_functions(&self) -> &Vec<fn(X)->X> {
        &self.activations
    }

    pub fn get_random_activation_function(&self) -> fn(X)->X {
        self.activations[self.activations.len().random()]
    }

    pub fn new_cppn(&mut self) -> CPPN<X> {
        let (cppn, inno) = CPPN::new(self.input_size, self.output_size, self.get_global_innovation_no());
        self.set_global_innovation_no(inno);
        cppn
    }

    pub fn new_cppns(&mut self, num: usize) -> Vec<CPPN<X>> {
        let mut vec = Vec::with_capacity(num);
        if num==0{return vec;}
        let inno = self.get_global_innovation_no();
        let (cppn, new_inno) = CPPN::new(self.input_size, self.output_size, inno);
        vec.push(cppn);
        for _ in 1..num {
            // All the created CPPNs share the same innovation numbers
            // but only differ in randomly initialised weights
            let (cppn, updated_inno) = CPPN::new(self.input_size, self.output_size, inno);
            assert_eq!(new_inno, updated_inno);
            vec.push(cppn);
        }
        self.set_global_innovation_no(new_inno);
        vec
    }

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    pub fn get_output_size(&self) -> usize {
        self.output_size
    }
    pub fn new_default(input_size: usize, output_size: usize) -> Self {
        Self::new(Vec::from_iter(X::ALL_ACT_FN.iter().cloned()), input_size, output_size)
    }
    pub fn new(activations: Vec<fn(X)->X>, input_size: usize, output_size: usize) -> Self {
        Self { global_innovation_no: 0, activations, input_size, output_size }
    }

    pub fn set_global_innovation_no(&mut self, val: usize) {
        assert!(val>=self.global_innovation_no,"Was {} and new value is {}",self.global_innovation_no, val);
        self.global_innovation_no = val;
    }

    pub fn get_global_innovation_no(&self) -> usize {
        self.global_innovation_no
    }

    pub fn activation_functions_len(&self) -> usize {
        self.activations.len()
    }

    pub fn get_activation_function(&self, i: usize) -> fn(X)->X {
        self.activations[i]
    }

    pub fn get_input_slice_mut<'a>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[..self.input_size]
    }

    pub fn get_input_slice<'a>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[..self.input_size]
    }

    pub fn get_output_slice_mut<'a>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[self.input_size..self.input_size + self.output_size]
    }

    pub fn get_output_slice<'a>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[self.input_size..self.input_size + self.output_size]
    }
    /**returns true if successful*/
    pub fn add_connection_if_possible(&mut self, cppn: &mut CPPN<X>, from: usize, to: usize) -> bool {
        let inno = self.get_global_innovation_no();
        let new_inno = cppn.add_connection_if_possible(from, to, X::random(), inno);
        self.set_global_innovation_no(new_inno);
        new_inno != inno
    }
    /**Returns index of the newly created node*/
    pub fn add_node(&mut self, cppn: &mut CPPN<X>, edge_index: usize) {
        let inno = self.get_global_innovation_no();
        let af = self.get_random_activation_function();
        let new_inno = cppn.add_node(edge_index, af, inno);
        debug_assert!(inno<new_inno,"Was {} and updated to {}",inno,new_inno);
        self.set_global_innovation_no(new_inno);
    }

    /**Randomly adds a new connection, but may fail if such change would result in recurrent
    neural net instead of feed-forward one (so acyclicity must be preserved). Returns true if successfully
    added a new edge*/
    pub fn add_random_connection(&mut self, cppn: &mut CPPN<X>) -> bool {
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        let b = self.add_connection_if_possible(cppn, cppn.get_random_node(), cppn.get_random_node());
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        b
    }

    pub fn add_random_node(&mut self, cppn: &mut CPPN<X>) {
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        self.add_node(cppn, cppn.get_random_edge());
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
    }

    pub fn make_output_buffer<'x,  I: Iterator<Item=&'x CPPN<X>>>(&'x self, population: I) -> Option<Vec<X>> {
        population.map(CPPN::node_count).max().map(|m| vec![X::zero(); m])
    }

    pub fn mutate(&mut self, cppn: &mut CPPN<X>,
                          node_insertion_prob: f32,
                          edge_insertion_prob: f32,
                          activation_fn_mutation_prob: f32,
                          weight_mutation_prob: f32,
                          enable_edge_prob: f32,
                          disable_edge_prob: f32) {
        let was_acyclic = cppn.is_acyclic();
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        if f32::random() < node_insertion_prob {
            self.add_random_node(cppn)
        }
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after add random node");
        if f32::random() < edge_insertion_prob {
            self.add_random_connection(cppn);
        }
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after add random connection");
        for edge_index in 0..cppn.edge_count() {
            if f32::random() < weight_mutation_prob {
                cppn.set_weight(edge_index, cppn.get_weight(edge_index).random_walk())
            }
            if cppn.is_enabled(edge_index){
                if f32::random() < disable_edge_prob {
                    cppn.set_enabled(edge_index, false);
                }
            }else{
                if f32::random() < enable_edge_prob {
                    cppn.set_enabled(edge_index, true);
                }
            }
        }
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        for node_index in 0..cppn.node_count() {
            if f32::random() < activation_fn_mutation_prob {
                cppn.set_activation(node_index, self.get_random_activation_function());
            }
        }
        debug_assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after mutate");
        debug_assert_eq!(was_acyclic, cppn.is_acyclic());
    }
}
