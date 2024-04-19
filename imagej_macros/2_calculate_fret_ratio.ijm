tot_pos=28;
data_dir="D:\\Experiments\\Temperature_project\\04-12-24\\Processed_Data";
save_dir="D:\\Experiments\\Temperature_project\\04-12-24\\Ratio";

setBatchMode(true);
File.makeDirectory(save_dir);
for (pos=0; pos<tot_pos; pos=pos+1){
	// Open FRET images
	File.openSequence(data_dir+"/Pos"+pos+"/", "virtual filter=FRET");
	rename("Pos"+pos+"_FRET");
	// Open CFP images
	File.openSequence(data_dir+"/Pos"+pos+"/", "virtual filter=CFP");
	rename("Pos"+pos+"_CFP");
	// Divide FRET by CFP
	imageCalculator("Divide create 32-bit stack", "Pos"+pos+"_FRET","Pos"+pos+"_CFP");
	selectWindow("Result of Pos"+pos+"_FRET");
	rename("Pos"+pos+"_Ratio");
	saveAs("Tiff", save_dir+"\\Pos"+pos+"_Ratio.tif");
	close("*");
}