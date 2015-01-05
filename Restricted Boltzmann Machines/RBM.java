/*************************************************************************   
    > File Name: RBM.java 
    > Author: SongLee   
    > E-mail: lisong.shine@qq.com   
    > Created Time: 2014年1月5日 星期五 11时15分50秒   
    > Personal Blog: http://songlee24.github.com   
 ************************************************************************/ 
package com.test.rbm;

import java.util.Random;

public class RBM {
	
	private int round;  // the round of training
	private int n;      // the number of visible units
	private int m;      // the number of hidden units
	private double[][] W;       // the weight matrix
	private double[] v_bias;     // the bias of visible layer
	private double[] h_bias;     // the bias of hidden layer
	public Random random = new Random(123);
	
	/**
	 * Constructor
	 */
	public RBM(int round, int n, int m, double[][] W, double[] v_bias, double[] h_bias) {
		this.round = round;
		this.n = n;
		this.m = m;
		this.W = W;
		this.v_bias = v_bias;
		this.h_bias = h_bias;
	}
	
	
	/**
	 * Overloading Constructor
	 */
	public RBM(int round, int n, int m)
	{
		this.round = round;
		this.n = n;
		this.m = m;
		
		/* initialize W randomly */
		W = new double[n][m];
		double x = 1.0 / n;
		for(int i=0; i<n; ++i)
			for(int j=0; j<m; ++j)
				W[i][j] = uniform(-x, x);
		
		/* initialize v_bias randomly */
		v_bias = new double[n];
		for(int i=0; i<n; ++i)
			v_bias[i] = 0;
		
		/* initialize h_bias randomly */
		h_bias = new double[m];
		for(int j=0; j<m; ++j)
			h_bias[j] = 0;
	}
	
	
	/**
	 * k-step contrastive divergence
	 */
	public void cd(int[] v_state, double learning_rate, int k)
	{
		int[] sample_h = new int[m];
		int[] sample_v = new int[n];
		
		sample_h_given_v(v_state, sample_h);
		sample_v_given_h(sample_h, sample_v);
	}
	
	
	/**
	 * Updating the hidden states Given visible states
	 */
	public void sample_h_given_v(int[] v, int[] sample) {
		for(int j=0; j<m; ++j) {			
			double sum = 0.0;
			for(int i=0; i<n; ++i) {
				sum += v[i]*W[i][j];
			}
			sum += h_bias[j];
			/* hidden unit j is set to 1 with the probability */
			double probability = sigmoid(sum);
			sample[j] = generateBinary(probability);
		}
	}
	
	
	/**
	 * Updating the visible states Given hidden states
	 */
	public void sample_v_given_h(int[] h, int[] sample) {
		for(int i=0; i<n; ++i) {			
			double sum = 0.0;
			for(int j=0; j<m; ++j) {
				sum += h[j]*W[i][j];
			}
			sum += v_bias[i];
			/* visible unit i is set to 1 with the probability */
			double probability = sigmoid(sum);
			sample[i] = generateBinary(probability);
		}
	}
	
	
	/**
	 * sigmoid(x) = 1/(1 + exp(-x))
	 */
	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
	
	
	/**
	 * generate a binary number (0 or 1)
	 */
	private int generateBinary(double p) {
		if(p < 0 || p > 1)
			return 0;
		double r = random.nextDouble();
		return r<p ? 1 : 0;
	}
	
	
	/**
	 * uniform function for initialize W, v_bias and h_bias.
	 */
	private double uniform(double min, double max) {
		return random.nextDouble() * (max - min) + min;
	}
	
	
	/**
	 * The entrance of the application
	 */
	public static void main(String[] args) {
		int[] training_data = {1, 1, 1, 0, 0, 0};
		RBM rbm = new RBM(2, 6, 4);
		rbm.cd(training_data, 0.1, 1);
	}
}
