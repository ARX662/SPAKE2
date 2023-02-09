import uk.ac.ic.doc.jpair.api.Pairing;
import uk.ac.ic.doc.jpair.pairing.EllipticCurve;
import uk.ac.ic.doc.jpair.pairing.Point;
import uk.ac.ic.doc.jpair.pairing.Predefined;

import java.util.Random;

    public class  Generators {
    // this is the starting point
    static Pairing e = Predefined.nssTate();

    //get P, which is a random point in group G1
    static Point P= e.RandomPointInG1(new Random());


        //DDH group // CDH//ECC group
        // motivation of the solution
        //implementation
        //security analysis
        // efficiency complexity
        // future development  pre computation resistance

    //the curve on which G1 is defined
     static EllipticCurve g1 = e.getCurve();
    // M will be another point used for calculating  of T which will be sent to bob






}
