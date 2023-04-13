import uk.ac.ic.doc.jpair.pairing.BigInt;
import uk.ac.ic.doc.jpair.pairing.Point;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Random;
public class Admin {
    public static void main(String[] args) throws Exception {
        //uncomment print statements to check for correctness of the output.
        // Wait for Password.txt to be created
        while (!new File("Password.txt").exists()) {
            Thread.sleep(1000);
        }
        // reads Password from the user
        System.out.println("Admin: Decrypting...");
        File Password_decrypt = new File("Password.txt");
        AESEncryption.decrypt(Password_decrypt,"Password_decrypted.txt");

        BufferedReader passwordreader = new BufferedReader(new FileReader("Password_decrypted.txt"));
        String passwordB = passwordreader.readLine();
        passwordreader.close();
        BigInt wB = new BigInt(passwordB);
        String wBS = wB.toString();


        BigInt n = new BigInt(160, new SecureRandom());

        //Creating a sub Generators N
        Point N = Generators.g1.multiply(Generators.P,n);
        // wN and send to client
        Point wN = Generators.g1.multiply(N, wB);

        // g^b = bP
        //Point aP is computed over G1
        BigInt b = new BigInt(160, new SecureRandom());
        Point Y = Generators.g1.multiply(Generators.P, b);

        //SB will be sent to alice
        Point SB = Generators.g1.add(Y,wN);
        String SSB = SB.toString();


        System.out.println("Admin: Encrypting data. Computing and sending data to Client...");

        //send  data to client
        BigInt xB = (BigInt) SB.getX();
        BigInt yB = (BigInt) SB.getY();
        String xStringB = xB.toString();
        String yStringB = yB.toString();
        String pointStringS = xStringB + "," + yStringB;
        FileOutputStream fosS = new FileOutputStream("S.txt");
        DataOutputStream outStreamS = new DataOutputStream(new BufferedOutputStream(fosS));
        outStreamS.write(pointStringS.getBytes(StandardCharsets.UTF_8));
        outStreamS.close();
        // send the Generators to client
        BigInt Nx = (BigInt) wN.getX();
        BigInt Ny = (BigInt) wN.getY();
        String NStringx = Nx.toString();
        String NStringy = Ny.toString();
        String pointStringN = NStringx + "," + NStringy;
        FileOutputStream fosN = new FileOutputStream("wN.txt");
        DataOutputStream outStreamN = new DataOutputStream(new BufferedOutputStream(fosN));
        outStreamN.write(pointStringN.getBytes(StandardCharsets.UTF_8));
        outStreamN.close();
        // Encrypt file
        File S_text = new File("S.txt");
        File wN_text = new File("wN.txt");
        AESEncryption.encrypt(S_text);
        AESEncryption.encrypt(wN_text);
        // waiting for client to send data
        // Wait for T.txt and wM.txt to be created
        while (!new File("T.txt").exists() || !new File("wM.txt").exists() ) {
            Thread.sleep(1000);
        }
        System.out.println("Admin: Decrypting data from client...");
        // decrypt
        File T_decrypt = new File("T.txt");
        File wM_decrypt = new File("wM.txt");

        AESEncryption.decrypt(T_decrypt,"T_decrypted.txt");
        AESEncryption.decrypt(wM_decrypt,"wM_decrypted.txt");

        // read data from client
        BufferedReader readerT = new BufferedReader(new FileReader("T_decrypted.txt"));
        String StringT = readerT.readLine();
        readerT.close();

        String [] coordinatesB = StringT.split(",");
        BigInt xR = new BigInt(coordinatesB[0]);
        BigInt yR = new BigInt(coordinatesB[1]);
        Point TB = new Point(xR, yR);
        String TSB =  TB.toString();



        // read the sub Generators from client
        BufferedReader readerM = new BufferedReader(new FileReader("wM_decrypted.txt"));
        String StringwM = readerM.readLine();
        readerM.close();

        String [] coordinatesM = StringwM.split(",");
        BigInt xM = new BigInt(coordinatesM[0]);
        BigInt yM = new BigInt(coordinatesM[1]);
        Point wM = new Point(xM,yM);



        System.out.println("Admin: Computing Key and Generating Session Key...");
        //Bob will compute the following : b(T-wM) =abP
        Point Kb = Generators.g1.multiply(Generators.g1.subtract(TB,wM),b);
        //Converting to String
        String KbS = Kb.toString();


        // hash of values sent and available to Bob(Hashing)

        String FinalStringBob = TSB+SSB+KbS+wBS;
        MessageDigest messageB = MessageDigest.getInstance("SHA-256");
        byte[] hashB = messageB.digest(FinalStringBob.getBytes(StandardCharsets.UTF_8));
        BigInt sB = new BigInt(1, hashB);
        String Sb = sB.toString();


        // saving the Session key for Client to check
        FileOutputStream Session = new FileOutputStream("session_keyAdmin.txt");
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Session));
        out.write(Sb.getBytes(StandardCharsets.UTF_8));
        out.close();
        File SessionA = new File("session_keyAdmin.txt");
        AESEncryption.encrypt(SessionA);

        // waiting for client to send session key
        System.out.println("Admin: Key Generated. Waiting for Client Key...");
        // Wait for session_key.txt to be created
        while (!new File("session_key.txt").exists()  ) {
            Thread.sleep(1000);
        }
        File Session_decrypt = new File("session_key.txt");
        AESEncryption.decrypt( Session_decrypt,"ClientKey_decrypted.txt");
        BufferedReader session = new BufferedReader(new FileReader("ClientKey_decrypted.txt"));
        String SA = session.readLine();
        // session key
        BigInt sA = new BigInt(SA);
        session.close();

        // client checks if session key is valid
        if(sB.equals(sA)){
            System.out.println("Admin: Session Secured");

        }
        else{
            System.out.println("Admin: Session Incomplete!");

        }
        /*

        // password the admin has read from admin
        System.out.println("Password_Admin: " +wB);
        // Reading data sent from client wM and T
        System.out.println("T_Admin: " +TB);
        System.out.println("wM_Admin: " +wM);
        // S sent to client
        System.out.println("S_Admin: " +SB);
        // wN sent to client
        System.out.println("wN_Admin: " +wN);
        // key generated by admin
        System.out.println("Key_Admin: " +Kb);
         */


    }




}