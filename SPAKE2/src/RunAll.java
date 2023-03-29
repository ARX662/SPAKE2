public class RunAll   {
    public static void main(String[] args) throws InterruptedException {
        //calculate runtime
        Long start_time = System.nanoTime();
        // Run the client
        Thread thread1 = new Thread(() -> {
            try {
                Client.main(null);
            } catch (Exception e) {
                System.out.println("Client: Session timed out! "  +e);
                System.exit(0);
            }
        });
        // Run Admin
        Thread thread2 = new Thread(() -> {
            try {
                Admin.main(null);
            } catch (Exception e) {
                System.out.println("Admin: Session timed out!"  + e);
                System.exit(0);
            }
        });
        // Start the process.

        thread1.start();
        thread2.start();

        // Wait for them both to finish
        thread1.join();
        thread2.join();
        Long Time_Elapsed = System.nanoTime() - start_time;
        System.out.println("Total Execution time for both client and admin is: " +Time_Elapsed + " ns");

    }
}