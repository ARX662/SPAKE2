public class RunAll   {

    public static void main(String[] args) throws InterruptedException {


        // Run the client
        Thread thread1 = new Thread(() -> {
            try {
                Client.main(null);
            } catch (Exception e) {
               System.out.println("Client: Session timed out!");
                System.exit(0);
            }
        });
        // Run Admin
        Thread thread2 = new Thread(() -> {
            try {
                Admin.main(null);
            } catch (Exception e) {
                System.out.println("Admin: Session timed out!");
                System.exit(0);
            }
        });
        // Start the process.
                thread1.start();
                thread2.start();

        // Wait for them both to finish
                thread1.join();
                thread2.join();

    }
}
