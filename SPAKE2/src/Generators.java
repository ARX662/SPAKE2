import uk.ac.ic.doc.jpair.api.Pairing;
import uk.ac.ic.doc.jpair.pairing.BigInt;
import uk.ac.ic.doc.jpair.pairing.EllipticCurve;
import uk.ac.ic.doc.jpair.pairing.Point;
import uk.ac.ic.doc.jpair.pairing.Predefined;


import javax.lang.model.element.Element;
import java.util.Random;

public class  Generators {
    // this is the starting point
    static Pairing e = Predefined.nssTate();

    //get P, which is a random point in group G1
    static Point P = e.RandomPointInG1(new Random());


    //the curve on which G1 is defined
    static EllipticCurve g1 = e.getCurve();

}

