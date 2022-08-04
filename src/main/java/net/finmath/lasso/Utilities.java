/**
 * 
 */
package net.finmath.lasso;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Christian Fries
 */
public class Utilities {

	/**
	 * Multiplication of two matrices.
	 *
	 * @param left The matrix A.
	 * @param right The matrix B
	 * @return product The matrix product of A*B (if suitable)
	 */
	public static double[][] mult(final double[][] left, final double[][] right) {
		return (new Array2DRowRealMatrix(left)).multiply(new Array2DRowRealMatrix(right)).getData();
	}

	/**
	 * Multiplication of matrices and vector
	 *
	 * @param matrix The matrix A.
	 * @param vector The vector v
	 * @return product The vector product of A*v (if suitable)
	 */
	public static double[] mult(final double[][] matrix, final double[] vector) {
		return (new Array2DRowRealMatrix(matrix)).operate(vector);
	}

	public static double[] subArray(final double[] array, int startIndexInclusive, int endIndexExclusive) {
		return ArrayUtils.subarray(array, startIndexInclusive, endIndexExclusive);
	}


	public static List<List<String>> readCSVTableWithHeaders(String pathToFile) throws IOException {
		return readCSVTableWithHeaders(pathToFile, CSVFormat.DEFAULT);

	}

	public static List<List<String>> readCSVTableWithHeaders(String pathToFile, CSVFormat format) throws IOException {
		Reader in = new FileReader(pathToFile);
		Iterable<CSVRecord> records = format.parse(in);		
		
		List<List<String>> table = new ArrayList<List<String>>();
		for (CSVRecord record : records) {
			List<String> row = new ArrayList<String>();
			for(String value : record) {
				row.add(value);
			}
			table.add(row);
		}
		
		return table;
	}
}
