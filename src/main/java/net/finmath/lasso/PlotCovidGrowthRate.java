/**
 * 
 */
package net.finmath.lasso;

import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import net.finmath.plots.Plots;

/**
 * @author fries
 *
 */
public class PlotCovidGrowthRate {

	/**
	 * 1: BW, 2: BY, ..., 7: HE, ..., 10: NRW, ... 16: TH, 17: DE (all)
	 */
	private static final int country = 17;
	private static final int periodLength = 14;

	public static void main(String[] args) {

		String countryName;
		List<Double> values = new ArrayList<>();

		try {
			List<List<String>> table = Utilities.readCSVTableWithHeaders("data/covid-de-20200506-20220805.csv");

			countryName = table.get(0).get(country);
			
			// Read data
			for(int rowIndex = 1; rowIndex < table.size(); rowIndex++) {
				List<String> row = table.get(rowIndex);

				LocalDate date = LocalDate.parse(row.get(0), DateTimeFormatter.ISO_LOCAL_DATE);

				Double value = Double.parseDouble(row.get(country));
				values.add(value);

				System.out.println("Read " + date);
			}
			System.out.println();
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}

		/**
		 * Calculate running sum over 7 days.
		 */
		List<Double> runningSums = new ArrayList<>();
		double runningSum = 0;
		for(int i=0; i<7; i++) {
			runningSum += values.get(i);
		}
		runningSums.add(runningSum);

		for(int i=7; i<values.size(); i++) {
			runningSum += values.get(i)-values.get(i-7);
			runningSums.add(runningSum);
		}

		/**
		 * Calculate grows rate
		 */
		List<Double> days = new ArrayList<>();
		List<Double> growthRates = new ArrayList<>();
		for(int i=periodLength+1; i<runningSums.size(); i++) {
			days.add((double)i);
			growthRates.add(Math.log(runningSums.get(i)/runningSums.get(i-periodLength))/periodLength);
		}
		
		Plots.createScatter(days, growthRates, 7, days.size()+7, 3)
		.setTitle("Covid Growth Rate for " + countryName)
		.setXAxisLabel("day")
		.setYAxisLabel("log growth rate")
		.show();
	}
}
