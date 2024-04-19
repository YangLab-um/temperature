// User-defined variables
tot_pos=28;
data_dir="D:\\Experiments\\Temperature_project\\04-12-24\\Raw_Data";
save_dir="D:\\Experiments\\Temperature_project\\04-12-24\\Processed_Data"; // Creates this directory
first_to_remove=550; // If not removing anything, put total_frames + 1 here
last_to_remove=601;

// Internal variables
resize_depth=first_to_remove-1;
starting_x=newArray(tot_pos);
starting_y=newArray(tot_pos);
ending_x=newArray(tot_pos);
ending_y=newArray(tot_pos);
line_widths=newArray(tot_pos);


// Loop for region of interest selection
for (pos=0; pos<tot_pos; pos=pos+1){
	// Open bright-field image, resize it and ask user to select region of interest
	open(data_dir+"/Pos"+pos+"/img_000000000_0-BF_000.tif");
	run("Size...", "width=1024 height=1024 depth=1 constrain average interpolation=Bilinear");
	waitForUser("Region of interest selection", "Please use thick line selection to mark tube on the bright field channel. Recommended thickness 740.");
	getLine(x1, y1, x2, y2, width);
	if(width < 2){
		waitForUser("Plase select a line width greater than 2. Double click on straight line selection to select line width");
		getLine(x1, y1, x2, y2, width);
	}
	if(y1 < y2){
		waitForUser("Plase start your line selection at the BOTTOM of the image");
		getLine(x1, y1, x2, y2, width);
	}
	// Throw an error if nothing was selected
	if (x1==-1){exit("This macro requires a straight line selection");}
	// Store data	
	starting_x[pos] = x1;
	starting_y[pos] = y1;
	ending_x[pos] = x2;
	ending_y[pos] = y2;
	line_widths[pos] = width;
	close();
}
waitForUser("Move to image processing", "Region of interest selection has finished. Click OK to proceed with image processing");

// Loop for image processing
File.makeDirectory(save_dir);
for (pos=0; pos<tot_pos; pos=pos+1){
	File.makeDirectory(save_dir+"/Pos"+pos);
	// Bright-field
	File.openSequence(data_dir+"/Pos"+pos+"/", "virtual filter=BF");
	rename("Pos"+pos+"_BF");
	run("Slice Remover", "first="+first_to_remove+" last="+last_to_remove+" increment=1");
	run("Size...", "width=1024 height=1024 depth="+resize_depth+" constrain average interpolation=Bilinear");
	makeLine(starting_x[pos], starting_y[pos], ending_x[pos], ending_y[pos]);
	run("Straighten...", "title=img_BF_ line="+line_widths[pos]+" process");
	run("Rotate 90 Degrees Left");
	run("Image Sequence... ", "dir="+save_dir+"/Pos"+pos+"/ format=TIFF digits=3");
	close();
	selectWindow("Pos"+pos+"_BF");
	close();
	// CFP
	File.openSequence(data_dir+"/Pos"+pos+"/", "virtual filter=CFP");
	rename("Pos"+pos+"_CFP");
	run("Slice Remover", "first="+first_to_remove+" last="+last_to_remove+" increment=1");
	run("Size...", "width=1024 height=1024 depth="+resize_depth+" constrain average interpolation=Bilinear");
	makeLine(starting_x[pos], starting_y[pos], ending_x[pos], ending_y[pos]);
	run("Straighten...", "title=img_CFP_ line="+line_widths[pos]+" process");
	run("Rotate 90 Degrees Left");
	File.makeDirectory(save_dir+"/Pos"+pos);
	run("Image Sequence... ", "dir="+save_dir+"/Pos"+pos+"/ format=TIFF digits=3");
	close();
	selectWindow("Pos"+pos+"_CFP");
	close();
	// FRET
	File.openSequence(data_dir+"/Pos"+pos+"/", "virtual filter=FRET");
	rename("Pos"+pos+"_FRET");
	run("Slice Remover", "first="+first_to_remove+" last="+last_to_remove+" increment=1");
	run("Size...", "width=1024 height=1024 depth="+resize_depth+" constrain average interpolation=Bilinear");
	makeLine(starting_x[pos], starting_y[pos], ending_x[pos], ending_y[pos]);
	run("Straighten...", "title=img_FRET_ line="+line_widths[pos]+" process");
	run("Rotate 90 Degrees Left");
	run("Image Sequence... ", "dir="+save_dir+"/Pos"+pos+"/ format=TIFF digits=3");
	close();
	selectWindow("Pos"+pos+"_FRET");
	close();
}