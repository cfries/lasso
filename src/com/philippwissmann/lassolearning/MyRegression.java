/**
 * 
 */
package com.philippwissmann.lassolearning;


/**
 * This is a class that reads a dataset for a regression.
 * @author phili
 *
 */
public class MyRegression {
	
	/**
	 * dimensionality of the predictors
	 */
	private int dimensionality;
	
	/**
	 * number of observations
	 */
	private int numberOfObservations;
	
	/**
	 * center of the response
	 */
	private double centerOfTheResponse;
	
	/**
	 * scaling factor of the response
	 */
	private double scaleOfTheResponse;
	
	/**
	 * saves the centered and scaled response 
	 */
	private double[] centeredScaledResponse;
	
	/**
	 * saves the center vector of the features in the design matrix
	 */
	private double[] centerVectorOfTheDesignMatrix;
	
	/**
	 * saves the scaling vector of the features in the design matrix
	 */
	private double[] scalingVectorOfTheDesignMatrix;
	
	/**
	 * saves the design matrix of the predictor
	 */
	private double[][] designMatrix;
	
	/**
	 * Constructor.
	 * @param predictor is the predictor matrix of the data set.
     * @param response is the response vector of the data set.
     * @param featureCentering is the boolean to center the predictor.
     * @param featureStandardization is the boolean to standardize the predictor.
	 */
	public MyRegression(double[][] predictor, double[] response, boolean featureCentering, boolean featureStandardization) {
    	
    	this.dimensionality = predictor[0].length; // set the dimensionality
        this.numberOfObservations = response.length; // set the number of Observations
        
        this.centerOfTheResponse = findCenter(response); // set the center of the response vector
        this.scaleOfTheResponse = findStandardizationFactor(response); // set the scale factor of the response vector
        
        for (int i=0; i<numberOfObservations; i++) { // let's save the centered and scaled response vector
        	this.centeredScaledResponse[i] = (response[i] - centerOfTheResponse) / scaleOfTheResponse;
        }
         
        boolean isAlreadyTheDesignMatrix = true; // let's save the design matrix and be flexible if the input is the design matrix or just the predictor matrix
        predictorOrDesignMatrixloop:
        for (int i=0; i<numberOfObservations; i++) { // and let's assume that if the 0-th feature of the predictor is 1 for all observations, then it is already a design matrix
        	if (predictor[i][0] != 1) {
        		isAlreadyTheDesignMatrix = false;
        		break predictorOrDesignMatrixloop;
        	}
        }
        if (isAlreadyTheDesignMatrix) {
        	for (int i=0; i<numberOfObservations; i++) { // loop over observations
				for (int j=0; j<dimensionality; j++) { // loop over feature
					designMatrix[i][j] = predictor[i][j];
				}
        	}
        } else {
        	dimensionality++;
        	for (int i=0; i<numberOfObservations; i++) { // loop over observations
        		designMatrix[i][0] = 1.0;
				for (int j=1; j<dimensionality; j++) { // loop over feature
					designMatrix[i][j] = predictor[i][j];
				}
        	}
        }
        
        if (featureCentering) { // if featureCentering is true, then we center the feature vectors
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { // we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = predictor[i][j];
        		}
        		centerVectorOfTheDesignMatrix[j] = findCenter(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { // centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] - centerVectorOfTheDesignMatrix[j];
        		}
        	}
        	
        }
        
        if (featureStandardization) { // if featureCentering is true, then we center the feature vectors
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { // we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = predictor[i][j];
        		}
        		scalingVectorOfTheDesignMatrix[j] = findStandardizationFactor(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { // centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] / scalingVectorOfTheDesignMatrix[j];
        		}
        	}
        }
        
        
        System.out.println("Which algorithm do you want to use to train Lasso?");        
        
        

    }
	
	/**
     * method to find the center of a given vector via Kahan summation
     * @param originalResponse
     * @return sum(originalResponse) / originalResponse.length
     */
    private static double findCenter(double[] originalVector) {   	
    	// let's sum the response values via the Kahan sum algorithm since potential datasets can be very big
    	double theTheoreticalSum = 0.0;
		double error = 0.0;
		for (int i =0; i < originalVector.length; i++) {
			double value = (double)originalVector[i]-error;
			double newSum = theTheoreticalSum + value;
			error = (newSum - theTheoreticalSum) - value;
			theTheoreticalSum = newSum;
		}
		return theTheoreticalSum / originalVector.length;
    }  
    
    /**
     * method to find the mean squared sum of a given vector via Kahan summation
     * @param originalResponse
     * @return sum(originalResponse) / originalResponse.length
     */
    private static double findStandardizationFactor(double[] originalVector) {
    	// let's sum the squared response values via the Kahan sum algorithm since potential datasets can be very big
    	double theTheoreticalSum = 0.0;
		double error = 0.0;
		for (int i =0; i < originalVector.length; i++) {
			double value = (double)originalVector[i] * (double)originalVector[i] - error;
			double newSum = theTheoreticalSum + value;
			error = (newSum - theTheoreticalSum) - value;
			theTheoreticalSum = newSum;
		}
		return Math.sqrt(theTheoreticalSum) / originalVector.length;
    }
	
}
