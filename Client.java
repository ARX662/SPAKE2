import uk.ac.ic.doc.jpair.pairing.BigInt;
import uk.ac.ic.doc.jpair.pairing.Point;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Random;
import java.util.Scanner;

public class Client   {

    public static void main(String[] args) throws Exception {

        Scanner Enter = new Scanner(System.in);  // Create a Scanner object
        System.out.println("Enter password:");
        String password = Enter.nextLine();  // Read user input
        //hashing the password with SHA-256 ,then converting it to a bigINT
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] digest = md.digest(password.getBytes(StandardCharsets.UTF_8));
        BigInt w = new BigInt(1, digest);
        String wS = w.toString();

        // saving the hash of the password to a file
        FileOutputStream fos = new FileOutputStream("Password.txt");
        DataOutputStream outStream = new DataOutputStream(new BufferedOutputStream(fos));
        outStream.write(wS.getBytes(StandardCharsets.UTF_8));
        outStream.close();

        BigInt m = new BigInt(160,new Random());
        // Creating a sub Generator wM
         Point M = Generators.g1.multiply(Generators.P,m);
        //Creating a sub Generators wM and send to admin
        Point wM = Generators.g1.multiply(M, w);



        // Alice wants to compute g^a where a is known to Alice and secret to the rest of the world
        // g = P
        // here we take the Generators to be P.
        // the secret random a
        //a is a 160-bit random integer
        BigInt a = new BigInt(160, new Random());
        // g^a = aP
        //Point aP is computed over G1
        Point X = Generators.g1.multiply(Generators.P, a);
        // T will be sent to Bob
        Point T = Generators.g1.add(X, wM);
        String TS = T.toString();// converting to String


        //Data being sent to admin
        BigInt x = (BigInt) T.getX();
        BigInt y = (BigInt) T.getY();
        String xString = x.toString();
        String yString = y.toString();
        String pointStringA = xString + "," + yString;
        FileOutputStream fosT = new FileOutputStream("T.txt");
        DataOutputStream outStreamT = new DataOutputStream(new BufferedOutputStream(fosT));
        outStreamT.write(pointStringA.getBytes(StandardCharsets.UTF_8));
        outStreamT.close();
        // wM  send to admin
        BigInt Mx = (BigInt) wM.getX();
        BigInt My = (BigInt) wM.getY();
        String MStringx = Mx.toString();
        String MStringy = My.toString();
        String pointStringN = MStringx + "," + MStringy;
        FileOutputStream fosM = new FileOutputStream("wM.txt");
        DataOutputStream outStreamM = new DataOutputStream(new BufferedOutputStream(fosM));
        outStreamM.write(pointStringN.getBytes(StandardCharsets.UTF_8));
        outStreamM.close();
        // Encrypt file
        File T_text = new File("T.txt");
        File wM_text = new File("wM.txt");
        AESEncryption.encrypt(w.toByteArray(),wM_text);
        AESEncryption.encrypt(w.toByteArray(),T_text);

        // wait for client to send data
        try {
            // Sleep for 5 seconds
            Thread.sleep(7500);
        } catch (InterruptedException e) {
            // Do nothing
        }
        // decrypt
        File SA_decrypt = new File("S.txt");
        File wNA_decrypt = new File("wN.txt");
        AESEncryption.decrypt(w.toByteArray(),SA_decrypt);
        AESEncryption.decrypt(w.toByteArray(),wNA_decrypt);
        //Read Data from admin
        BufferedReader reader = new BufferedReader(new FileReader("S.txt"));
        String pointR = reader.readLine();

        reader.close();
        String[] coordinates = pointR.split(",");
        BigInt AxR = new BigInt(coordinates[0]);
        BigInt AyR = new BigInt(coordinates[1]);
        Point S = new Point(AxR, AyR);
        String SS = S.toString();

        // read the sub Generators from admin
        BufferedReader readerN = new BufferedReader(new FileReader("wN.txt"));
        String StringwN = readerN.readLine();
        readerN.close();

        String[] coordinatesN = StringwN.split(",");
        BigInt xN = new BigInt(coordinatesN[0]);
        BigInt yN = new BigInt(coordinatesN[1]);
        Point wN = new Point(xN, yN);
       
/////////////////////////////////////////////////////////////////////////////


        //Alice will compute the following : a(S-wN) =abP
        Point Y = Generators.g1.subtract(S, wN);
        Point Ka = Generators.g1.multiply(Y, a);
        //Converting to String
        String KaS = Ka.toString();
        


        // hash of values sent and available to alice(Hashing)
        String FinalStringAlice = TS + SS + KaS + wS;
        MessageDigest messageA = MessageDigest.getInstance("SHA-256");
        byte[] hashA = messageA.digest(FinalStringAlice.getBytes(StandardCharsets.UTF_8));
        BigInt sA = new BigInt(1, hashA);
        String Sa = sA.toString();

        // saving the Session key for admin to check
        FileOutputStream SA = new FileOutputStream("session_key.txt");
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(SA));
        out.write(Sa.getBytes(StandardCharsets.UTF_8));
        out.close();


        // waiting for admin to send session key
        try {
            // Sleep for 10 seconds
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            // Do nothing
        }
        BufferedReader session = new BufferedReader(new FileReader("session_keyAdmin.txt"));
        String SB = session.readLine();
        // session key
        BigInt sB = new BigInt(SB);

        session.close();

        if(sB.equals(sA)){
            System.out.println("Client:Session Secured");

        }
        else{
            System.out.println("Session Incomplete!");

        }

    }




}
