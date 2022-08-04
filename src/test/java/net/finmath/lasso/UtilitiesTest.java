/**
 * 
 */
package net.finmath.lasso;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.Date;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * @author fries
 *
 */
class UtilitiesTest {

	@Test
	void testCSVRead() {
		
		System.out.println("Reading the CSV table to List<List<String>>");
		try {
			List<List<String>> table = Utilities.readCSVTableWithHeaders("data/covid-de-20200506-20220805.csv");
			
			// Print the first n entries
			for(int rowIndex = 0; rowIndex < 10; rowIndex++) {
				List<String> row = table.get(rowIndex);
				System.out.println(row);
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}

		System.out.println();
	}

	@Test
	void testCSVParsing() {
		
		System.out.println("Parsing the CSV table to Java types");
		
		try {
			List<List<String>> table = Utilities.readCSVTableWithHeaders("data/covid-de-20200506-20220805.csv");

			// Print the first n entries, skip first row (header)
			for(int rowIndex = 1; rowIndex < 10; rowIndex++) {
				List<String> row = table.get(rowIndex);
				
				LocalDate date = LocalDate.parse(row.get(0), DateTimeFormatter.ISO_LOCAL_DATE);
				System.out.print(date);
				
				for(int colIndex = 1; colIndex < row.size(); colIndex++) {
					Double value = Double.parseDouble(row.get(colIndex));
					System.out.print("\t");
					System.out.print(value);
				}
				System.out.println();
			}
			System.out.println();
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
}
