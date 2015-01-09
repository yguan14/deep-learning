/*************************************************************************   
    > File Name: RBM.java 
    > Author: SongLee   
    > E-mail: lisong.shine@qq.com   
    > Created Time: Jan 4th, 2015 Sunday 11:15:50   
    > Personal Blog: http://songlee24.github.com   
 ************************************************************************/ 
package com.test.rbm;

import java.util.Random;

public class RBM {
	
	private int n;      // the number of visible units
	private int m;      // the number of hidden units
	private double[][] W;       // the weight matrix
	private double[] v_bias;     // the bias of visible layer
	private double[] h_bias;     // the bias of hidden layer
	public Random random = new Random(123);
	
	/**
	 * Constructor
	 */
	public RBM(int n, int m, double[][] W, double[] v_bias, double[] h_bias) {
		this.n = n;
		this.m = m;
		this.W = W;
		this.v_bias = v_bias;
		this.h_bias = h_bias;
	}
	
	
	/**
	 * Overloading Constructor
	 */
	public RBM(int n, int m)
	{
		this.n = n;
		this.m = m;
		
		/* initialize W randomly */
		W = new double[n][m];
		double x = 1.0 / n;
		for(int i=0; i<n; ++i)
			for(int j=0; j<m; ++j)
				W[i][j] = uniform(-x, x);
		
		/* initialize v_bias */
		v_bias = new double[n];
		for(int i=0; i<n; ++i)
			v_bias[i] = 0;
		
		/* initialize h_bias */
		h_bias = new double[m];
		for(int j=0; j<m; ++j)
			h_bias[j] = 0;
	}
	
	
	/**
	 * k-step contrastive divergence
	 */
	public void cd(int[] v1, double lr, int k)
	{
		int[] sample_h1 = new int[m];
		int[] sample_v2 = new int[n];
		int[] sample_h2 = new int[m];
		double[] probability_h1 = new double[m];
		double[] probability_v2 = new double[n];
		double[] probability_h2 = new double[m];
		
		sample_h_given_v(v1, sample_h1, probability_h1);
		
		for(int step=0; step<k; ++step) {
			if(step == 0) {
				sample_v_given_h(sample_h1, sample_v2, probability_v2);
				sample_h_given_v(sample_v2, sample_h2, probability_h2);
			} else {
				sample_v_given_h(sample_h2, sample_v2, probability_v2);
				sample_h_given_v(sample_v2, sample_h2, probability_h2);
			}
		}
		
		/* update the weights and biases */
		for(int j=0; j<m; ++j) {
			for(int i=0; i<n; ++i) {
				W[i][j] += lr*(probability_h1[j]*v1[i]-probability_h2[j]*sample_v2[i]);
			}
			h_bias[j] += lr*(probability_h1[j]-probability_h2[j]);
		}
		
		for(int i=0; i<n; ++i)
			v_bias[i] += lr*(v1[i]-sample_v2[i]);
		
		System.out.println(getMarginalProbability(v1));
	}
	
	/**
	 * sum up the marginal probability
	 */
	public double getMarginalProbability(int[] v) {
		double sum = 0.0;
		for(int j=0; j<m; ++j) {
			for(int i=0; i<n; ++i) {
				sum -= v[i]*v_bias[i];
			}
		
			double temp = 0.0;
			for(int i=0; i<n; ++i)
				temp -= v[i] * W[i][j];
			temp -= h_bias[j];
		
			sum += Math.log1p(Math.pow(Math.E, temp));
		}
		return sum;
	}
	
	
	private double energy(int[] v, int[] h) {
		double sum = 0.0;
		for(int i=0; i<n; ++i) {
			for(int j=0; j<m; ++j)
				sum -= v[i]*h[j]*W[i][j];
			sum -= v[i]*v_bias[i];
		}
		
		for(int j=0; j<m; ++j)
			sum -= h[j]*h_bias[j];
		
		return sum;
	}
	
	/**
	 * Updating the hidden states Given visible states
	 */
	public void sample_h_given_v(int[] v, int[] sample, double[] p) {
		for(int j=0; j<m; ++j) {			
			double sum = 0.0;
			for(int i=0; i<n; ++i) {
				sum += v[i]*W[i][j];
			}
			sum += h_bias[j];
			/* hidden unit j is set to 1 with the probability */
			double probability = sigmoid(sum);
			p[j] = probability;
			sample[j] = generateBinary(probability);
		}
	}
	
	
	/**
	 * Updating the visible states Given hidden states
	 */
	public void sample_v_given_h(int[] h, int[] sample, double[] p) {
		for(int i=0; i<n; ++i) {			
			double sum = 0.0;
			for(int j=0; j<m; ++j) {
				sum += h[j]*W[i][j];
			}
			sum += v_bias[i];
			/* visible unit i is set to 1 with the probability */
			double probability = sigmoid(sum);
			p[i] = probability;
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
		// training data
//		int[][] train_data = {
//				{1, 1, 1, 0, 0, 0},
//				{1, 0, 1, 0, 0, 0},
//				{1, 1, 1, 0, 0, 0},
//				{0, 0, 1, 1, 1, 0},
//				{0, 0, 1, 0, 1, 0},
//				{0, 0, 1, 1, 1, 0}
//		};
//		
//		double learning_rate = 0.1;
//		int training_epoch = 1000; 
//		int round = 2;
//		int n = 6;
//		int m = 6;
//		int k = 1;
//		RBM rbm = new RBM(round, n, m);
//		
//		// train
//		for(int i=0; i<training_epoch; ++i)
//			//for(int j=0; j<6; ++j)
//				rbm.cd(train_data[1], learning_rate, k);
//		
//		// test data
//		int[][] test_data = {
//			{1, 1, 0, 0, 0, 0},
//			{0, 0, 0, 1, 1, 0}
//		};
		int[] v = {1, 0};
		double[][] W = {{0, 0},{0, 0}};
		double[] a = {0.5, 0.25};
		double[] b = {0, 0};
		int n = 2;
		int m = 2;
		int k = 1;
		RBM rbm = new RBM(n, m, W, a, b);
		//System.out.println(rbm.getMarginalProbability(v));
		int[] h1 = new int[m];
		double[] p = new double[m];
		rbm.sample_h_given_v(v, h1, p);
		
		for(int i=0; i<m; ++i)
			System.out.print(p[i]+": " +h1[i]+ " \n");
		System.out.println();
	}
}
